package nlp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	genai "github.com/google/generative-ai-go/genai"
)

// Response schema from the LLM. Keep it tiny and strict.
type Rewrite struct {
	Primary      string   `json:"primary"`
	Alternatives []string `json:"alternatives"`
}

// Extract plain text from a GenerateContentResponse (legacy SDK).
func extractText(resp *genai.GenerateContentResponse) string {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0] == nil {
		return ""
	}
	c := resp.Candidates[0].Content
	if c == nil || len(c.Parts) == 0 {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		b.WriteString(fmt.Sprint(p))
	}
	return b.String()
}

// Normalize output: trim, dedupe, drop empties, limit alts.
func normalize(r Rewrite) Rewrite {
	seen := map[string]struct{}{}
	clean := func(s string) (string, bool) {
		s = strings.TrimSpace(s)
		if s == "" {
			return "", false
		}
		if _, ok := seen[strings.ToLower(s)]; ok {
			return "", false
		}
		seen[strings.ToLower(s)] = struct{}{}
		return s, true
	}
	out := Rewrite{}
	if p, ok := clean(r.Primary); ok {
		out.Primary = p
	}
	for _, a := range r.Alternatives {
		if v, ok := clean(a); ok {
			out.Alternatives = append(out.Alternatives, v)
			if len(out.Alternatives) >= 3 {
				break
			}
		}
	}
	return out
}

// RewriteQuery calls Gemini to spell-correct/normalize the query.
// modelName e.g. "gemini-1.5-flash" or "gemini-1.5-pro".
func RewriteQuery(ctx context.Context, gm *genai.GenerativeModel, raw string) (Rewrite, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Rewrite{}, errors.New("empty query")
	}

	// System-style concise instruction to force strict JSON.
	prompt := `
You are a query rewriter for an e-commerce product search. 
Tasks:
1) Fix spelling mistakes and normalize the query while preserving user intent.
2) Keep brand/model/series tokens (e.g., "iPhone 14 Pro", "Galaxy S23") intact if obvious.
3) Return STRICT JSON ONLY with this schema (no markdown, no prose):

{
  "primary": "<one corrected query string>",
  "alternatives": ["<alt1>", "<alt2>"]
}

Guidelines:
- Avoid adding new intent or extra adjectives.
- Prefer common brand spellings (e.g., "samsung", "google", "iphone").
- If the input is already clean, return it unchanged as "primary".
- Provide up to 2 short alternatives (synonyms, close spellings) or an empty list.
`

	inp := fmt.Sprintf("Input: %q", raw)
	resp, err := gm.GenerateContent(ctx, genai.Text(prompt), genai.Text(inp))
	if err != nil {
		return Rewrite{}, err
	}

	txt := extractText(resp)
	dec := json.NewDecoder(strings.NewReader(txt))
	dec.DisallowUnknownFields()

	var r Rewrite
	if err := dec.Decode(&r); err != nil {
		// Fallback: use original when model returns bad JSON
		return Rewrite{Primary: raw, Alternatives: nil}, nil
	}

	r = normalize(r)
	if r.Primary == "" {
		// Fallback if model blanked primary
		r.Primary = raw
	}
	return r, nil
}
