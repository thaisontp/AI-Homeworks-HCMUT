import math

# Initial Values
i1 = 0.05
i2 = 0.10

target_o1 = 0.01
target_o2 = 0.99

w1 = 0.15; w2 = 0.20
w3 = 0.25; w4 = 0.30
w5 = 0.40; w6 = 0.45
w7 = 0.50; w8 = 0.55

b1 = 0.35
b2 = 0.60

learning_rate = 0.5 

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(out):
    return out * (1 - out)

#EARLY STOPPING SETUP
max_epochs = 10000
patience = 10           
patience_counter = 0
previous_error = float('inf')
epsilon = 1e-6          

print("START TRAINING: 2-2-2 NETWORK \n")

#TRAINING LOOP (EPOCHS)
for epoch in range(1, max_epochs + 1):
    
    #Forward Pass
    net_h1 = (w1 * i1) + (w2 * i2) + b1
    out_h1 = sigmoid(net_h1)

    net_h2 = (w3 * i1) + (w4 * i2) + b1
    out_h2 = sigmoid(net_h2)

    net_o1 = (w5 * out_h1) + (w6 * out_h2) + b2
    out_o1 = sigmoid(net_o1)

    net_o2 = (w7 * out_h1) + (w8 * out_h2) + b2
    out_o2 = sigmoid(net_o2)

    # Total Error
    total_error = 0.5 * ((target_o1 - out_o1)**2 + (target_o2 - out_o2)**2)

    # --- Error Backpropagation ---
    # Output Layer
    dE_dout_o1 = out_o1 - target_o1 
    delta_o1 = dE_dout_o1 * sigmoid_derivative(out_o1)
    dE_dw5 = delta_o1 * out_h1
    dE_dw6 = delta_o1 * out_h2

    dE_dout_o2 = out_o2 - target_o2
    delta_o2 = dE_dout_o2 * sigmoid_derivative(out_o2)
    dE_dw7 = delta_o2 * out_h1
    dE_dw8 = delta_o2 * out_h2

    dE_db2 = delta_o1 + delta_o2

    # Hidden Layer
    dE_dout_h1 = (delta_o1 * w5) + (delta_o2 * w7)
    delta_h1 = dE_dout_h1 * sigmoid_derivative(out_h1)
    dE_dw1 = delta_h1 * i1
    dE_dw2 = delta_h1 * i2

    dE_dout_h2 = (delta_o1 * w6) + (delta_o2 * w8)
    delta_h2 = dE_dout_h2 * sigmoid_derivative(out_h2)
    dE_dw3 = delta_h2 * i1
    dE_dw4 = delta_h2 * i2

    dE_db1 = delta_h1 + delta_h2

    w1 -= learning_rate * dE_dw1
    w2 -= learning_rate * dE_dw2
    w3 -= learning_rate * dE_dw3
    w4 -= learning_rate * dE_dw4
    w5 -= learning_rate * dE_dw5
    w6 -= learning_rate * dE_dw6
    w7 -= learning_rate * dE_dw7
    w8 -= learning_rate * dE_dw8

    b1 -= learning_rate * dE_db1
    b2 -= learning_rate * dE_db2

    #EPOCH 1
    if epoch == 1:
        print("[STEP 1] CALCULATION (EPOCH 1)")
        print(f"Total Error: {total_error:.5f}")
        print(f"out_o1: {out_o1:.5f} (Target: {target_o1})")
        print(f"out_o2: {out_o2:.5f} (Target: {target_o2})")
        print("-" * 115 + "\n")
        print("[STEP 2] FULL EPOCHS (TRAINING PROGRESS)")

    print(f"Epoch {epoch:>4} | E: {total_error:.6f} | W_ih: [{w1:.3f}, {w2:.3f}, {w3:.3f}, {w4:.3f}] | b_h: [{b1:.3f}] | W_ho: [{w5:.3f}, {w6:.3f}, {w7:.3f}, {w8:.3f}] | b_o: [{b2:.3f}]")

    #EARLY STOPPING
    error_diff = abs(previous_error - total_error)
    if error_diff < epsilon:
        patience_counter += 1
    else:
        patience_counter = 0 
        
    if patience_counter >= patience:
        print("-" * 115)
        print(f">>> EARLY STOPPING ACTIVATED:")
        print(f">>> Error has not changed significantly for {patience} consecutive epochs!")
        print(f">>> STOPPING TRAINING AT EPOCH {epoch}")
        break
        
    previous_error = total_error

#FINAL RESULTS
print("-" * 115)
print("--- Final Prediction Results ---")
print(f"Final Error: {total_error:.6f}")
print(f"Prediction o1: {out_o1:.5f} (Target: {target_o1})")
print(f"Prediction o2: {out_o2:.5f} (Target: {target_o2})")
print("\n--- Final Converged Parameters ---")
print(f"w1: {w1:.5f} | w2: {w2:.5f} | w3: {w3:.5f} | w4: {w4:.5f}")
print(f"w5: {w5:.5f} | w6: {w6:.5f} | w7: {w7:.5f} | w8: {w8:.5f}")
print(f"b1: {b1:.5f} | b2: {b2:.5f}")
