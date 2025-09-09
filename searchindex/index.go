package searchindex

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	genai "github.com/google/generative-ai-go/genai"
	"github.com/xrash/smetrics"
)

type Product struct {
	ID          uint
	SellerID    uint
	CategoryID  uint
	Title       string
	Description string
	Brand       string
	Status      int
	Score       int
}

type productDoc struct {
	P          Product
	Embedding  []float32
	SearchText string
}

type SearchResult struct {
	Product Product `json:"product"`
	Score   float64 `json:"score"`
	Why     struct {
		Semantic float64 `json:"semantic"`
		Fuzzy    float64 `json:"fuzzy"`
	} `json:"why"`
}

type Index struct {
	em             *genai.EmbeddingModel
	modelName      string
	semanticWeight float64
	fuzzyWeight    float64

	mu   sync.RWMutex
	docs []productDoc
}

func New(ctx context.Context, client *genai.Client, modelName string, semanticWeight, fuzzyWeight float64) *Index {
	return &Index{
		em:             client.EmbeddingModel(modelName),
		modelName:      modelName,
		semanticWeight: semanticWeight,
		fuzzyWeight:    fuzzyWeight,
	}
}

func (ix *Index) Rebuild(ctx context.Context, products []Product) error {
	var docs []productDoc
	for _, p := range products {
		joined := strings.TrimSpace(strings.Join([]string{p.Title, p.Brand, p.Description}, " "))
		if joined == "" {
			continue
		}
		resp, err := ix.em.EmbedContent(ctx, genai.Text(joined))
		if err != nil {
			return fmt.Errorf("embed product %d: %w", p.ID, err)
		}
		docs = append(docs, productDoc{
			P:          p,
			Embedding:  resp.Embedding.Values,
			SearchText: joined,
		})
	}
	ix.mu.Lock()
	ix.docs = docs
	ix.mu.Unlock()
	return nil
}

func (ix *Index) Search(ctx context.Context, query string, topK int) ([]SearchResult, error) {
	q := strings.TrimSpace(query)
	if q == "" {
		return []SearchResult{}, nil
	}

	qResp, err := ix.em.EmbedContent(ctx, genai.Text(q))
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	qVec := qResp.Embedding.Values

	ix.mu.RLock()
	defer ix.mu.RUnlock()

	results := make([]SearchResult, 0, len(ix.docs))
	for _, d := range ix.docs {
		sem := cosine(qVec, d.Embedding)
		fuz := max3(
			jaroWinkler(q, d.P.Title),
			jaroWinkler(q, d.P.Brand),
			jaroWinkler(q, d.P.Description),
		)
		score := ix.semanticWeight*sem + ix.fuzzyWeight*fuz

		var r SearchResult
		r.Product = d.P
		r.Score = score
		r.Why.Semantic = sem
		r.Why.Fuzzy = fuz
		results = append(results, r)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if topK > 0 && topK < len(results) {
		results = results[:topK]
	}
	return results, nil
}

func cosine(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i] * b[i])
		na += float64(a[i] * a[i])
		nb += float64(b[i] * b[i])
	}
	den := math.Sqrt(na) * math.Sqrt(nb)
	if den == 0 {
		return 0
	}
	return dot / den
}

func jaroWinkler(a, b string) float64 {
	a = strings.ToLower(strings.TrimSpace(a))
	b = strings.ToLower(strings.TrimSpace(b))
	return smetrics.JaroWinkler(a, b, 0.7, 4)
}

func max3(a, b, c float64) float64 { return math.Max(a, math.Max(b, c)) }
