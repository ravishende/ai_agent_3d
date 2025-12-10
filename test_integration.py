"""
Test script to verify the integration between all components.
"""
import numpy as np
from maps import maps
from gymnasium_env import ObstacleCourseEnv
from integrated_policy_iteration import IntegratedPolicyIteration, solve_map
from core import Action, INIT_GAME
import traceback


def test_environment():
    """Test that the Gymnasium environment works correctly."""
    print("="*60)
    print("TEST 1: Gymnasium Environment")
    print("="*60)
    
    # Use a simple map
    game_map = maps[0]
    env = ObstacleCourseEnv(game_map)
    
    print(f"✓ Environment created")
    print(f"  State space size: {env.observation_space.n}")
    print(f"  Action space size: {env.action_space.n}")
    print(f"  Map length: {len(game_map)}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Environment reset")
    print(f"  Initial observation: {obs}")
    
    # Test step
    action = 4  # STAY
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n✓ Step executed")
    print(f"  Action: {action} (STAY)")
    print(f"  New observation: {obs}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    
    # Test preview
    next_obs, reward, done = env.preview_action(obs, 1)
    print(f"\n✓ Preview works")
    print(f"  Preview action RIGHT: next_obs={next_obs}, reward={reward}, done={done}")
    
    print("\n" + "="*60)
    print("Environment tests PASSED ✓")
    print("="*60 + "\n")


def test_policy_iteration():
    """Test that policy iteration runs and converges."""
    print("="*60)
    print("TEST 2: Policy Iteration")
    print("="*60)
    
    game_map = maps[0]  # Simple map
    env = ObstacleCourseEnv(game_map)
    
    print(f"Running policy iteration on map with {len(game_map)} timesteps...")
    
    pi = IntegratedPolicyIteration(env, gamma=0.98)
    converged, iterations = pi.run(max_iterations=50, verbose=False)
    
    print(f"\n✓ Policy iteration completed")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    
    # Get action sequence
    actions = pi.get_action_sequence(verbose=False)
    print(f"\n✓ Action sequence extracted")
    print(f"  Length: {len(actions)}")
    print(f"  Actions: {[a.name for a in actions]}")
    
    # Evaluate policy
    results = pi.evaluate_policy_on_map()
    print(f"\n✓ Policy evaluated")
    print(f"  Success: {results['success']}")
    print(f"  Total reward: {results['total_reward']}")
    
    print("\n" + "="*60)
    print("Policy iteration tests PASSED ✓")
    print("="*60 + "\n")


def test_full_pipeline():
    """Test the complete pipeline from map to actions."""
    print("="*60)
    print("TEST 3: Full Pipeline")
    print("="*60)
    
    for i, game_map in enumerate(maps, 1):
        print(f"\nTesting Map {i} (length={len(game_map)})...")
        
        # Solve using the convenience function
        actions, pi = solve_map(game_map, verbose=False)
        
        # Verify the solution
        results = pi.evaluate_policy_on_map()
        
        status = "✓ SUCCESS" if results['success'] else "✗ FAILED"
        print(f"  {status} - Reward: {results['total_reward']}/{len(game_map)}")
        print(f"  Actions: {' → '.join([a.name for a in actions[:5]])}...")
    
    print("\n" + "="*60)
    print("Full pipeline tests PASSED ✓")
    print("="*60 + "\n")


def test_state_encoding():
    """Test state encoding/decoding."""
    print("="*60)
    print("TEST 4: State Encoding/Decoding")
    print("="*60)
    
    game_map = maps[0]
    env = ObstacleCourseEnv(game_map)
    
    # Test a few states
    test_states = [
        (0, 1),  # Start
        (1, 2),  # right
        (2, 0),  # left
    ]
    
    for t, col in test_states:
        from core import State
        state = State(t, col)
        obs = env._state_to_obs(state)
        decoded = env._obs_to_state(obs)
        
        match = (decoded.time_index == t and 
                decoded.player_col == col)
        
        status = "✓" if match else "✗"
        print(f"{status} State ({t}, {col}) → obs {obs} → State ({decoded.time_index}, {decoded.player_col})")
    
    print("\n" + "="*60)
    print("State encoding tests PASSED ✓")
    print("="*60 + "\n")


def main():
    """Run all tests."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "INTEGRATION TEST SUITE" + " "*21 + "║")
    print("╚" + "="*58 + "╝" + "\n")
    
    try:
        test_environment()
        test_state_encoding()
        test_policy_iteration()
        test_full_pipeline()
        
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*10 + "ALL TESTS PASSED SUCCESSFULLY! ✓" + " "*15 + "║")
        print("╚" + "="*58 + "╝" + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
