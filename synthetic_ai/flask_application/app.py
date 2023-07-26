from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import dill

# Initialize Flask application
app = Flask(__name__)

# Load the model using dill.load
with open('model.pkl', 'rb') as f:
    loaded_model = dill.load(f)

# Set the device for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)
loaded_model.eval()

# Function to generate text from the model
def generate_text(model, block_size=8):
    generated_text = ''
    out = []
    context = [0] * block_size
    itos={1: ' ', 2: ',', 3: '.', 4: '0', 5: '1', 6: '2', 7: '3', 8: '4', 9: '5', 10: '6', 11: '7', 12: '8', 13: '9', 14: 'a', 15: 'b', 16: 'c', 17: 'd', 18: 'e', 19: 'f', 20: 'g', 21: 'h', 22: 'i', 23: 'j', 24: 'k', 25: 'l', 26: 'm', 27: 'n', 28: 'o', 29: 'p', 30: 'r', 31: 's', 32: 't', 33: 'u', 34: 'v', 35: 'w', 36: 'x', 37: 'y', 38: 'z', 0: '*'}
     
    with torch.no_grad():
        while True:
            # Convert the context to a tensor and transfer it to the device
            context_tensor = torch.tensor([context], device=device)

            # Forward pass the neural net
            logits = model(context_tensor)
            probs = F.softmax(logits, dim=1)

            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()

            # Shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)

            # If we sample the special '.' token, break
            if ix == 0:
                break

    generated_text = ''.join(itos[i] for i in out)
    return generated_text.replace("*", "")

# Route to the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and display the predictions
@app.route('/predict', methods=['POST'])
def predict():
    num_predictions = int(request.form['num_predictions'])  # Number of predictions to display
    predictions = [generate_text(loaded_model) for _ in range(num_predictions)]
    return render_template('predictions.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

