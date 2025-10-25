/*
This package implements Markov chain-based prioritization for ffuf.

The Markov chain logic works as follows:

1. State representation: ⟨code_class, size_bucket, depth⟩
   - code_class: "2xx", "3xx", "4xx", "5xx" 
   - size_bucket: quantized response body length
   - depth: path depth

2. Actions: the fuzz tokens/words being tested

3. Transitions: S_t --(action)--> S_{t+1}, observed from ffuf responses

4. Rewards: 1 for "useful" responses (non-404; preference 2xx/3xx/401/403, "new" content),
             0 for baseline 404 responses

The goal is to learn P(S'|S, action) and prioritize inputs that are most likely 
to escape the "404 attractor" and return meaningful responses.
*/
package markov

// The Markov chain implementation provides a way to prioritize fuzzing inputs
// based on learned patterns from previous responses, helping to escape 404 responses
// and find meaningful resources more efficiently.