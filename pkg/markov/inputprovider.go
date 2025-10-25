package markov

import (
	"sync"
)

// InputProvider interface - matches the ffuf input provider interface
type InputProvider interface {
	Next() bool
	Value() map[string][]byte
	Position() int
	SetPosition(int)
	Keywords() []string
	ActivateKeywords([]string)
	Reset()
	Total() int
}

// MarkovInputProvider wraps the original InputProvider with Markov chain logic
type MarkovInputProvider struct {
	OriginalProvider InputProvider
	MarkovChain      *MarkovChain
	previousInputs   map[string][]byte
	currentBatch     []map[string][]byte
	currentIndex     int
	batchSize        int
	baselineState    State
	baselineSizeHash string
	depth            int
	mutex            sync.Mutex
}

// NewMarkovInputProvider creates a new input provider with Markov chain logic
func NewMarkovInputProvider(original InputProvider, baselineState State, baselineSizeHash string, depth int) *MarkovInputProvider {
	return &MarkovInputProvider{
		OriginalProvider: original,
		MarkovChain:      NewMarkovChain(),
		previousInputs:   make(map[string][]byte),
		currentBatch:     make([]map[string][]byte, 0),
		currentIndex:     0,
		batchSize:        100, // Process inputs in batches to make better predictions
		baselineState:    baselineState,
		baselineSizeHash: baselineSizeHash,
		depth:            depth,
	}
}

// SetBaseline sets the baseline 404 state for comparison
func (mip *MarkovInputProvider) SetBaseline(baselineState State, baselineSizeHash string) {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()
	
	mip.baselineState = baselineState
	mip.baselineSizeHash = baselineSizeHash
}

// AddTransition adds a transition to the Markov chain based on a request-response cycle
func (mip *MarkovInputProvider) AddTransition(fromState State, action string, toState State, reward float64) {
	transition := Transition{
		FromState: fromState,
		Action:    Action{Token: action},
		ToState:   toState,
		Reward:    reward,
	}
	mip.MarkovChain.UpdateTransition(transition)
}

// Response structure - matches ffuf.Response fields we need
type Response struct {
	StatusCode    int64
	Headers       map[string][]string
	Data          []byte
	ContentLength int64
	ContentWords  int64
	ContentLines  int64
	ContentType   string
	Cancelled     bool
	Request       interface{} // simplified
	Raw           string
	ResultFile    string
	ScraperData   map[string][]string
	Duration      interface{} // time.Duration
	Timestamp     interface{} // time.Time
}

// UpdateWithResponse updates the Markov chain with a response
func (mip *MarkovInputProvider) UpdateWithResponse(inputs map[string][]byte, resp *Response) {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()

	// Create current state from response
	currentState := GetStateFromResponseFromResponseStruct(resp, mip.depth)
	
	// Get action (the value that was fuzzed, typically the FUZZ keyword)
	var actionValue string
	for kw, value := range inputs {
		// Look for FUZZ keyword which is standard in ffuf
		if kw == "FUZZ" {
			actionValue = string(value)
			break
		}
	}
	
	// Calculate reward based on the response
	reward := CalculateRewardFromResponseStruct(resp, mip.baselineState, mip.baselineSizeHash)
	
	// Create previous state from context (in a real implementation, we'd store this)
	// For now, we'll just use the baseline state as the previous state
	previousState := mip.baselineState
	
	// Add transition to Markov chain
	mip.AddTransition(previousState, actionValue, currentState, reward)
}

// GetStateFromResponseFromResponseStruct creates a state representation from our Response struct
func GetStateFromResponseFromResponseStruct(resp *Response, depth int) State {
	// Determine status code class
	var codeClass string
	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		codeClass = "2xx"
	case resp.StatusCode >= 300 && resp.StatusCode < 400:
		codeClass = "3xx"
	case resp.StatusCode >= 400 && resp.StatusCode < 500:
		codeClass = "4xx"
	case resp.StatusCode >= 500 && resp.StatusCode < 600:
		codeClass = "5xx"
	default:
		codeClass = "unknown"
	}

	// Quantize the content length into buckets
	sizeBucket := quantizeSize(resp.ContentLength)

	return State{
		CodeClass:  codeClass,
		SizeBucket: sizeBucket,
		Depth:      depth,
	}
}

// CalculateRewardFromResponseStruct determines the reward based on our Response struct
func CalculateRewardFromResponseStruct(resp *Response, baselineState State, baselineSizeHash string) float64 {
	// Base reward is 0
	reward := 0.0

	// If not a 404, increase reward
	if resp.StatusCode < 400 || resp.StatusCode >= 500 || resp.StatusCode == 401 || resp.StatusCode == 403 {
		// Non-404 responses get a base reward boost
		reward += 1.0
	}

	// If it's a 404, check if it's different from baseline
	if resp.StatusCode >= 400 && resp.StatusCode < 500 {
		currentState := GetStateFromResponseFromResponseStruct(resp, baselineState.Depth)
		
		// If this 404 has different characteristics than baseline, it might still be useful
		if currentState.Hash() != baselineState.Hash() {
			// Different size or content than baseline - potential for discovering something new
			currentSizeHash := GetSizeHash(resp.Data)
			if currentSizeHash != baselineSizeHash {
				reward += 0.5 // Somewhat interesting if content differs
			}
		}
	} else {
		// If we got a non-404 response, this is generally good
		reward += 1.0
	}

	// Prioritize success responses (2xx) and access forbidden (401, 403) over other status codes
	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		reward += 1.0 // Success responses are very valuable
	case resp.StatusCode == 401 || resp.StatusCode == 403:
		reward += 0.8 // Access forbidden suggests protected resource exists
	case resp.StatusCode == 404:
		// 404 gets no additional reward, but may have gotten some above if different from baseline
	case resp.StatusCode >= 500 && resp.StatusCode < 600:
		reward += 0.6 // Server errors might indicate something interesting
	}

	return reward
}

// RefreshBatch refreshes the current batch with reordered inputs based on Markov predictions
func (mip *MarkovInputProvider) RefreshBatch() {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()

	// For now we'll just reset the original provider
	// Full implementation would require access to re-order the original wordlist
	// which would need deeper integration with the wordlist provider
	
	// Reset original provider to start fresh
	mip.OriginalProvider.Reset()
	mip.currentIndex = 0
	mip.currentBatch = make([]map[string][]byte, 0)

	// Fill the current batch with inputs
	for len(mip.currentBatch) < mip.batchSize && mip.OriginalProvider.Next() {
		inputs := make(map[string][]byte)
		originalInputs := mip.OriginalProvider.Value()
		for k, v := range originalInputs {
			inputs[k] = make([]byte, len(v))
			copy(inputs[k], v)
		}
		mip.currentBatch = append(mip.currentBatch, inputs)
	}

	// Reset position again to start fresh
	mip.OriginalProvider.Reset()
}

// Next moves to the next input in the current batch or gets a new batch based on Markov predictions
func (mip *MarkovInputProvider) Next() bool {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()

	// If we have more items in the current batch, use them
	if mip.currentIndex < len(mip.currentBatch) {
		mip.currentIndex++
		return true
	}

	// We need a new batch - first reset the index
	mip.currentIndex = 0

	// Refresh batch with Markov-driven reordering
	mip.RefreshBatch()

	// Check if we have items in the new batch
	if len(mip.currentBatch) > 0 {
		mip.currentIndex = 1 // Start at 1 since we return true and will call Value() next
		return true
	}

	// Fallback to original provider's Next method if batch refresh failed
	if mip.OriginalProvider != nil {
		return mip.OriginalProvider.Next()
	}

	return false
}

// Value returns the current input value
func (mip *MarkovInputProvider) Value() map[string][]byte {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()

	if mip.currentIndex > 0 && mip.currentIndex <= len(mip.currentBatch) {
		// Store the inputs for potential next transition
		inputs := mip.currentBatch[mip.currentIndex-1]
		for k, v := range inputs {
			mip.previousInputs[k] = make([]byte, len(v))
			copy(mip.previousInputs[k], v)
		}
		return inputs
	}

	// Fallback to original provider if no batch is available
	if mip.OriginalProvider != nil {
		originalInputs := mip.OriginalProvider.Value()
		result := make(map[string][]byte)
		for k, v := range originalInputs {
			result[k] = make([]byte, len(v))
			copy(result[k], v)
		}
		return result
	}

	// Return empty map as fallback
	return make(map[string][]byte)
}

// Position returns the current position
func (mip *MarkovInputProvider) Position() int {
	// We'll return the original provider's position to maintain compatibility
	if mip.OriginalProvider != nil {
		return mip.OriginalProvider.Position()
	}
	return 0
}

// SetPosition sets the position
func (mip *MarkovInputProvider) SetPosition(pos int) {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()
	
	if mip.OriginalProvider != nil {
		mip.OriginalProvider.SetPosition(pos)
	}
	mip.currentIndex = 0
	mip.currentBatch = make([]map[string][]byte, 0)
}

// Keywords returns the keywords
func (mip *MarkovInputProvider) Keywords() []string {
	if mip.OriginalProvider != nil {
		return mip.OriginalProvider.Keywords()
	}
	return []string{}
}

// ActivateKeywords activates keywords
func (mip *MarkovInputProvider) ActivateKeywords(kws []string) {
	if mip.OriginalProvider != nil {
		mip.OriginalProvider.ActivateKeywords(kws)
	}
}

// Reset resets the provider
func (mip *MarkovInputProvider) Reset() {
	mip.mutex.Lock()
	defer mip.mutex.Unlock()
	
	if mip.OriginalProvider != nil {
		mip.OriginalProvider.Reset()
	}
	mip.currentIndex = 0
	mip.currentBatch = make([]map[string][]byte, 0)
	mip.previousInputs = make(map[string][]byte)
}

// Total returns total number of inputs
func (mip *MarkovInputProvider) Total() int {
	if mip.OriginalProvider != nil {
		return mip.OriginalProvider.Total()
	}
	return 0
}