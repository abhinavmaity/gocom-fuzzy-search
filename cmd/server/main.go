package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"time"

	genai "github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"gocom_fuzzy_search/nlp"
	"google.golang.org/api/option"

	"gocom_fuzzy_search/models"      // your Product model
	"gocom_fuzzy_search/searchindex" // the engine above
)

func main() {
	_ = godotenv.Load() // ok if missing
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY is not set")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("genai.NewClient: %v", err)
	}
	defer client.Close()

	rewriterModelName := getenvDefault("QUERY_REWRITER_MODEL", "gemini-1.5-flash")
	rewriter := client.GenerativeModel(rewriterModelName)

	modelName := getenvDefault("EMBEDDING_MODEL", "text-embedding-004")
	semW := parseFloatDefault(os.Getenv("SEMANTIC_WEIGHT"), 0.70)
	fuzW := parseFloatDefault(os.Getenv("FUZZY_WEIGHT"), 0.30)

	ix := searchindex.New(ctx, client, modelName, semW, fuzW)

	// TODO: swap this with DB load via GORM (Marketplace DB)
	initial := []models.Product{
		{ID: 1, Title: "Apple iPhone 14 Pro", Brand: "Apple", Description: "6.1-inch, A16 Bionic, 48MP camera"},
		{ID: 2, Title: "Samsung Galaxy S23", Brand: "Samsung", Description: "Dynamic AMOLED 2X, Snapdragon"},
		{ID: 3, Title: "Google Pixel 8", Brand: "Google", Description: "Tensor G3, excellent camera"},
		{ID: 4, Title: "Nokia Lumia 950", Brand: "Nokia", Description: "PureView camera, AMOLED display"},
	}
	if err := ix.Rebuild(ctx, toIndexProducts(initial)); err != nil {
		log.Fatalf("initial rebuild: %v", err)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	// POST /reindex  (body: JSON array of products)
	mux.HandleFunc("/reindex", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var products []models.Product
		if err := json.NewDecoder(r.Body).Decode(&products); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		if err := ix.Rebuild(ctx, toIndexProducts(products)); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	// GET /search?q=...&topK=10
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		topK := parseIntDefault(r.URL.Query().Get("topK"), 10)

		ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
		defer cancel()

		// 1) Get rewrites from Gemini (spelling fixes, etc.)
		rw, err := nlp.RewriteQuery(ctx, rewriter, q)
		if err != nil {
			// On failure, just fall back to the raw query.
			rw = nlp.Rewrite{Primary: q}
		}

		// 2) Search for primary + alternatives and merge by best score
		type prodKey = uint
		best := map[prodKey]searchindex.SearchResult{}

		// helper to merge results by max score
		merge := func(list []searchindex.SearchResult) {
			for _, it := range list {
				id := it.Product.ID
				if prev, ok := best[id]; !ok || it.Score > prev.Score {
					best[id] = it
				}
			}
		}

		// primary
		resPrimary, err := ix.Search(ctx, rw.Primary, topK)
		if err == nil {
			merge(resPrimary)
		}

		// alternatives (cap at 2–3 from rewriter)
		for _, alt := range rw.Alternatives {
			resAlt, err := ix.Search(ctx, alt, topK)
			if err == nil {
				merge(resAlt)
			}
		}

		// 3) Flatten + sort
		out := make([]searchindex.SearchResult, 0, len(best))
		for _, v := range best {
			out = append(out, v)
		}
		sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
		if topK > 0 && topK < len(out) {
			out = out[:topK]
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			Query      string                     `json:"query"`
			Normalized nlp.Rewrite                `json:"normalized"`
			Results    []searchindex.SearchResult `json:"results"`
		}{
			Query:      q,
			Normalized: rw,
			Results:    out,
		})
	})

	addr := getenvDefault("ADDR", ":8080")
	log.Printf("fuzzy-search service listening on %s (model=%s, sem=%.2f, fuzzy=%.2f)",
		addr, modelName, semW, fuzW)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func getenvDefault(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
func parseIntDefault(s string, def int) int {
	if s == "" {
		return def
	}
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return def
}
func parseFloatDefault(s string, def float64) float64 {
	if s == "" {
		return def
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	return def
}

func toIndexProducts(ps []models.Product) []searchindex.Product {
	out := make([]searchindex.Product, 0, len(ps))
	for _, p := range ps {
		out = append(out, searchindex.Product{
			ID: p.ID, SellerID: p.SellerID, CategoryID: p.CategoryID,
			Title: p.Title, Description: p.Description, Brand: p.Brand,
			Status: p.Status, Score: p.Score,
		})
	}
	return out
}
