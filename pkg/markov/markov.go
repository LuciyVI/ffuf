package markov

import (
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"sync"
)

// State represents the state in our Markov chain
type State struct {
	CodeClass  string // "2xx", "3xx", "4xx", "5xx"
	SizeBucket string // quantized/rounded size for body length
	Depth      int    // depth of path
}

// Hash returns a hash representation of the state for use as map key
func (s State) Hash() string {
	return fmt.Sprintf("%s_%s_%d", s.CodeClass, s.SizeBucket, s.Depth)
}

// Action represents the fuzz token/word that was used
type Action struct {
	Token string // the actual fuzz word/token used
}

// Transition represents a transition from state S to state S' with an action
type Transition struct {
	FromState State
	Action    Action
	ToState   State
	Reward    float64
}

// MarkovChain holds the probability transition matrix and Q-values
type MarkovChain struct {
	// Q-values table: Q[state][action] = expected reward
	QTable map[string]map[string]float64

	// Transition counts: counts[state][action][next_state] = number of times this transition occurred
	TransitionCounts map[string]map[string]map[string]int

	// Action counts: counts[state][action] = total times action taken in state
	ActionCounts map[string]map[string]int

	// State visit counts: how many times each state has been visited
	StateCounts map[string]int

	// Available actions cache for each state
	AvailableActions map[string][]string

	// Mutex for thread safety
	mutex sync.RWMutex

	// Configurable parameters
	Alpha     float64 // Learning rate
	Gamma     float64 // Discount factor
	Epsilon   float64 // Exploration rate
	Threshold float64 // Minimum threshold to consider as improvement
}

// NewMarkovChain creates a new MarkovChain instance
func NewMarkovChain() *MarkovChain {
	return &MarkovChain{
		QTable:           make(map[string]map[string]float64),
		TransitionCounts: make(map[string]map[string]map[string]int),
		ActionCounts:     make(map[string]map[string]int),
		StateCounts:      make(map[string]int),
		AvailableActions: make(map[string][]string),
		Alpha:            0.1,  // Learning rate
		Gamma:            0.9,  // Discount factor
		Epsilon:          0.1,  // Exploration rate (10% of the time explore randomly)
		Threshold:        0.01, // Minimum threshold
	}
}



// QuantizeSize converts content length to a bucket representation (exported function)
func QuantizeSize(size int64) string {
	if size < 0 {
		size = 0
	}

	// Use logarithmic scale for large ranges, linear for small ranges
	if size < 100 {
		return fmt.Sprintf("%d", size/10*10) // Round to nearest 10
	} else if size < 1000 {
		return fmt.Sprintf("%d", size/100*100) // Round to nearest 100
	} else if size < 10000 {
		return fmt.Sprintf("%d", size/1000*1000) // Round to nearest 1000
	} else {
		return fmt.Sprintf("%d", size/10000*10000) // Round to nearest 10000
	}
}

// quantizeSize converts content length to a bucket representation
func quantizeSize(size int64) string {
	return QuantizeSize(size)
}

// GetSizeHash creates a hash of the response body content for more granular differentiation
func GetSizeHash(data []byte) string {
	if len(data) == 0 {
		return "0"
	}

	h := fnv.New64a()
	h.Write(data)
	return fmt.Sprintf("%x", h.Sum64())
}



// UpdateTransition updates the Q-value based on a state transition and reward
func (mc *MarkovChain) UpdateTransition(transition Transition) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	fromStateKey := transition.FromState.Hash()
	actionKey := transition.Action.Token
	toStateKey := transition.ToState.Hash()

	// Initialize maps if needed
	if _, exists := mc.QTable[fromStateKey]; !exists {
		mc.QTable[fromStateKey] = make(map[string]float64)
	}
	if _, exists := mc.TransitionCounts[fromStateKey]; !exists {
		mc.TransitionCounts[fromStateKey] = make(map[string]map[string]int)
	}
	if _, exists := mc.ActionCounts[fromStateKey]; !exists {
		mc.ActionCounts[fromStateKey] = make(map[string]int)
	}

	// Update action counts
	mc.ActionCounts[fromStateKey][actionKey]++

	// Update transition counts
	if _, exists := mc.TransitionCounts[fromStateKey][actionKey]; !exists {
		mc.TransitionCounts[fromStateKey][actionKey] = make(map[string]int)
	}
	mc.TransitionCounts[fromStateKey][actionKey][toStateKey]++

	// Update state visit counts
	mc.StateCounts[fromStateKey]++

	// Update Q-value using Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γmax(Q(s',a')) - Q(s,a)]
	currentQ := mc.QTable[fromStateKey][actionKey]
	
	// Find max Q-value for next state (if there are possible next actions)
	maxNextQ := 0.0
	if nextQs, exists := mc.QTable[toStateKey]; exists && len(nextQs) > 0 {
		for _, q := range nextQs {
			if q > maxNextQ {
				maxNextQ = q
			}
		}
	}

	// Q-learning update
	newQ := currentQ + mc.Alpha*(transition.Reward + mc.Gamma*maxNextQ - currentQ)
	mc.QTable[fromStateKey][actionKey] = newQ

	// Update available actions if this is a new action for this state
	found := false
	for _, existingAction := range mc.AvailableActions[fromStateKey] {
		if existingAction == actionKey {
			found = true
			break
		}
	}
	if !found {
		mc.AvailableActions[fromStateKey] = append(mc.AvailableActions[fromStateKey], actionKey)
	}
}

// GetBestActionsForState returns the top N actions for a given state, ordered by expected reward
func (mc *MarkovChain) GetBestActionsForState(state State, wordlist []string, n int) []string {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	stateKey := state.Hash()
	
	// If we don't have Q-values for this state, return the original wordlist
	if _, exists := mc.QTable[stateKey]; !exists {
		// If we've seen this state before but have no Q-values, return random
		if _, stateSeen := mc.StateCounts[stateKey]; stateSeen {
			return getRandomSubset(wordlist, n)
		}
		// Never seen this state, return original wordlist or random subset
		return getRandomSubset(wordlist, n)
	}

	// Create a list of (action, q-value) pairs
	type actionValue struct {
		action string
		value  float64
	}
	var actionValues []actionValue

	// Add all known actions for this state with their Q-values
	for action, qValue := range mc.QTable[stateKey] {
		// Only include actions that are actually in the wordlist
		if containsString(wordlist, action) {
			actionValues = append(actionValues, actionValue{action: action, value: qValue})
		}
	}

	// If no known actions are in the wordlist, return random subset
	if len(actionValues) == 0 {
		return getRandomSubset(wordlist, n)
	}

	// Sort by Q-value in descending order
	sort.Slice(actionValues, func(i, j int) bool {
		return actionValues[i].value > actionValues[j].value
	})

	// Return top N actions (or all if less than N)
	result := make([]string, 0, n)
	for i := 0; i < len(actionValues) && i < n; i++ {
		result = append(result, actionValues[i].action)
	}

	// If we have fewer than N actions, fill with remaining random words from wordlist
	// that aren't already in the result
	if len(result) < n {
		remaining := make([]string, 0)
		for _, word := range wordlist {
			if !containsString(result, word) {
				remaining = append(remaining, word)
			}
		}
		
		// Shuffle and add up to the remaining needed
		shuffleStrings(remaining)
		for i := 0; i < len(remaining) && len(result) < n; i++ {
			result = append(result, remaining[i])
		}
	}

	return result
}

// GetExpectedReward returns the expected reward for taking an action in a state
func (mc *MarkovChain) GetExpectedReward(state State, action string) float64 {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	stateKey := state.Hash()
	if qValues, exists := mc.QTable[stateKey]; exists {
		if qValue, actionExists := qValues[action]; actionExists {
			return qValue
		}
	}
	return 0.0 // Default reward if not known
}

// containsString checks if a string exists in a slice
func containsString(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// getRandomSubset returns a random subset of strings from the provided slice
func getRandomSubset(slice []string, n int) []string {
	if n >= len(slice) {
		return slice
	}

	// Create a copy and shuffle
	result := make([]string, len(slice))
	copy(result, slice)
	shuffleStrings(result)
	
	return result[:n]
}

// shuffleStrings shuffles a slice of strings in place (simple Fisher-Yates)
func shuffleStrings(slice []string) {
	// Use a simple deterministic shuffle for now
	// In a real implementation, we'd use crypto/rand or similar
	for i := len(slice) - 1; i > 0; i-- {
		// Simple pseudo-random swap (not cryptographically secure)
		j := int(math.Abs(float64(i*31))) % len(slice)
		slice[i], slice[j] = slice[j], slice[i]
	}
}