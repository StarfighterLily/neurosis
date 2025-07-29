import numpy as np
import random

class FeedforwardNetwork:
    def __init__(self, input_size=3, hidden_sizes=None, output_size=2, learning_rate=0.05):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [6]
        self.output_size = output_size
        self.learning_rate = learning_rate
        self._init_weights()
        self.last_input = None
        self.last_hidden = None
        self.last_output = None
        self.training_epochs = 0
        self.learning_progress = []
        self.weight_change_history = []
        self.cycle_count = 0
        self.max_cycles = 10
        self.current_pattern = 0
        self.training_data = []
        self.generate_training_patterns()

    def _init_weights(self):
        self.weights = []
        self.biases = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2/(layer_sizes[i]+layer_sizes[i+1]))
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        self.last_input = x
        self.last_hidden = activations[1:-1]
        self.last_output = activations[-1]
        return activations[-1]

    def backward(self, x, y):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        delta = activations[-1] - y
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = delta
        for l in range(2, len(self.weights)+1):
            z = zs[-l]
            sp = self.sigmoid_deriv(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            nabla_b[-l] = delta
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * nabla_w[i]
            self.biases[i] -= self.learning_rate * nabla_b[i]
        avg_change = np.mean([np.abs(nw).mean() for nw in nabla_w])
        self.weight_change_history.append(avg_change)
        if len(self.weight_change_history) > 50:
            self.weight_change_history.pop(0)

    def set_input_pattern(self, pattern):
        self.last_input = np.array(pattern).reshape(-1, 1)
    def get_output_pattern(self):
        if self.last_output is not None:
            return [int(round(v)) for v in self.last_output.flatten()]
        return [0] * self.output_size
    def update_network(self):
        if self.last_input is not None:
            self.forward(self.last_input)
        self.cycle_count += 1
    def train_network(self):
        current_input, desired_output = self.training_data[self.current_pattern]
        x = np.array(current_input).reshape(-1, 1)
        y = np.array(desired_output).reshape(-1, 1)
        self.forward(x)
        self.backward(x, y)
        error = np.abs(self.last_output - y).sum()
        self.learning_progress.append(error)
        if len(self.learning_progress) > 50:
            self.learning_progress.pop(0)
        self.training_epochs += 1
        #if self.weight_change_history:
            #print(f"Epoch {self.training_epochs}: Avg weight change = {self.weight_change_history[-1]:.6f}")
    def reset_network(self):
        self.last_input = None
        self.last_hidden = None
        self.last_output = None
        self.cycle_count = 0
        self.learning_progress.clear()
        self.weight_change_history.clear()
        # Do NOT reset self.training_epochs here; it should persist across patterns
        if self.training_data:
            current_input, _ = self.training_data[self.current_pattern]
            self.set_input_pattern(current_input)
    def generate_training_patterns(self):
        self.training_data = []
        for _ in range(4):
            input_pattern = [random.choice([0, 1]) for _ in range(self.input_size)]
            output_pattern = [random.choice([0, 1]) for _ in range(self.output_size)]
            self.training_data.append((input_pattern, output_pattern))
    def draw(self, screen, selected_layer=None):
        import pygame
        import numpy as np
        from ui_controls import small_font, font, GREEN, ORANGE, GRAY, WHITE, CYAN, RED, YELLOW
        num_layers = 1 + len(self.hidden_sizes) + 1
        layer_xs = np.linspace(200, 900, num_layers)
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        neurons = []
        for l, (x, n) in enumerate(zip(layer_xs, layer_sizes)):
            layer = []
            for i in range(n):
                y = 200 + i * 80
                color = GREEN if l == 0 and self.last_input is not None and self.last_input[i,0] > 0.5 else (
                    ORANGE if l == num_layers-1 and self.last_output is not None and self.last_output[i,0] > 0.5 else GRAY)
                pygame.draw.circle(screen, color, (int(x), y), 20)
                pygame.draw.circle(screen, WHITE, (int(x), y), 20, 2)
                text = small_font.render(str(i), True, WHITE)
                screen.blit(text, (int(x)-7, y-10))
                layer.append((int(x), y))
            neurons.append(layer)
        if selected_layer is not None and 0 < selected_layer < num_layers-1:
            x = int(layer_xs[selected_layer])
            y_top = 200 - 30
            y_bottom = 200 + (layer_sizes[selected_layer]-1)*80 + 30
            pygame.draw.rect(screen, CYAN, (x-40, y_top, 80, y_bottom-y_top), 3)
        for l in range(len(layer_sizes)-1):
            for i, (x0, y0) in enumerate(neurons[l]):
                for j, (x1, y1) in enumerate(neurons[l+1]):
                    w = self.weights[l][j, i]
                    color = GREEN if w > 0 else RED
                    thickness = max(1, int(abs(w) * 3))
                    pygame.draw.line(screen, color, (x0, y0), (x1, y1), thickness)
        info_text = f"Feedforward Network (Backprop)"
        text = font.render(info_text, True, WHITE)
        screen.blit(text, (20, 20))
        if self.training_data:
            current_input, desired_output = self.training_data[self.current_pattern]
            input_text = f"Input: {current_input}"
            desired_text = f"Desired: {desired_output}"
            actual_text = f"Actual: {self.get_output_pattern()}"
            text1 = small_font.render(input_text, True, WHITE)
            text2 = small_font.render(desired_text, True, GREEN)
            text3 = small_font.render(actual_text, True, YELLOW)
            screen.blit(text1, (20, 50))
            screen.blit(text2, (20, 70))
            screen.blit(text3, (20, 90))
        cycle_text = f"Cycles: {self.cycle_count}/{self.max_cycles}"
        epoch_text = f"Training Epochs: {self.training_epochs}"
        text = small_font.render(cycle_text, True, WHITE)
        text2 = small_font.render(epoch_text, True, WHITE)
        screen.blit(text, (20, 110))
        screen.blit(text2, (20, 130))
