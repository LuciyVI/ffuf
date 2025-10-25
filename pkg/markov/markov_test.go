package markov

import (
	"testing"
)

func TestStateHash(t *testing.T) {
	state1 := State{
		CodeClass:  "2xx",
		SizeBucket: "1000",
		Depth:      2,
	}
	
	state2 := State{
		CodeClass:  "2xx",
		SizeBucket: "1000",
		Depth:      2,
	}
	
	state3 := State{
		CodeClass:  "4xx",
		SizeBucket: "1000",
		Depth:      2,
	}
	
	if state1.Hash() != state2.Hash() {
		t.Errorf("Same states should have same hash: %s != %s", state1.Hash(), state2.Hash())
	}
	
	if state1.Hash() == state3.Hash() {
		t.Errorf("Different states should have different hashes: %s == %s", state1.Hash(), state3.Hash())
	}
}

func TestQuantizeSize(t *testing.T) {
	tests := []struct {
		input    int64
		expected string
	}{
		{0, "0"},
		{5, "0"},     // Rounds to nearest 10
		{15, "10"},   // Rounds to nearest 10
		{95, "90"},   // Rounds to nearest 10
		{105, "100"}, // Rounds to nearest 100
		{995, "900"}, // Rounds to nearest 100
		{1005, "1000"}, // Rounds to nearest 1000
		{9995, "9000"}, // Rounds to nearest 1000
		{10005, "10000"}, // Rounds to nearest 10000
	}
	
	for _, test := range tests {
		result := QuantizeSize(test.input)
		if result != test.expected {
			t.Errorf("QuantizeSize(%d) = %s; want %s", test.input, result, test.expected)
		}
	}
}

func TestMarkovChain(t *testing.T) {
	mc := NewMarkovChain()
	
	state1 := State{
		CodeClass:  "4xx",
		SizeBucket: "139",
		Depth:      1,
	}
	
	state2 := State{
		CodeClass:  "2xx",
		SizeBucket: "2000",
		Depth:      1,
	}
	
	action := "testfile.php"
	
	// Test initial state
	expected := mc.GetExpectedReward(state1, action)
	if expected != 0.0 {
		t.Errorf("Expected reward for new state/action should be 0.0, got %f", expected)
	}
	
	// Update with transition
	transition := Transition{
		FromState: state1,
		Action:    Action{Token: action},
		ToState:   state2,
		Reward:    1.0,
	}
	
	mc.UpdateTransition(transition)
	
	// Test that reward is now updated
	expected = mc.GetExpectedReward(state1, action)
	if expected <= 0.0 {
		t.Errorf("Expected reward should be positive after update, got %f", expected)
	}
}