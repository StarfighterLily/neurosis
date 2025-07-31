import pygame
import sys
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from network import FeedforwardNetwork
from network_rnn import RecurrentNetwork # Import the new RNN class
import ui_controls
from ui_controls import Slider, Checkbox

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Neurosis")

class Simulation:
    def __init__(self):
        # Network type management
        self.network_types = ['Feedforward', 'Recurrent']
        self.network_index = 0
        
        # Instantiate both network types
        self.feedforward = FeedforwardNetwork(input_size=3, hidden_sizes=[6], output_size=2, learning_rate=0.05)
        self.recurrent = RecurrentNetwork(input_size=3, hidden_size=6, output_size=2, learning_rate=0.01)

        self.selected_layer = 1
        self.running = True
        self.paused = True
        self.auto_train = True
        self.clock = pygame.time.Clock()
        self.speed_slider = Slider(20, WINDOW_HEIGHT - 100, 200, 10, 0.5, 20.0, 5.0)
        self.simulation_speed = 5.0
        self.max_speed_checkbox = Checkbox(250, WINDOW_HEIGHT - 100, 15, "Max Speed", False)

    def current_network(self):
        """Returns the currently active network object."""
        if self.network_types[self.network_index] == 'Feedforward':
            return self.feedforward
        else:
            return self.recurrent

    def draw_modern_controls(self):
        right_x1 = WINDOW_WIDTH - 260
        right_x2 = WINDOW_WIDTH - 130
        btn_w = 120
        btn_h = 30
        spacing = 10
        top_y = 60

        is_rnn = self.network_types[self.network_index] == 'Recurrent'

        # --- Neuron/Layer Controls ---
        add_neuron_btn = pygame.Rect(right_x1, top_y, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, add_neuron_btn)
        screen.blit(ui_controls.small_font.render("Add Neuron", True, ui_controls.BLACK), (add_neuron_btn.x + 10, add_neuron_btn.y + 5))
        
        rem_neuron_btn = pygame.Rect(right_x1, top_y + btn_h + spacing, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, rem_neuron_btn)
        screen.blit(ui_controls.small_font.render("Remove Neuron", True, ui_controls.BLACK), (rem_neuron_btn.x + 10, rem_neuron_btn.y + 5))
        
        add_layer_btn = pygame.Rect(right_x2, top_y, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.GRAY if is_rnn else ui_controls.LIGHT_BLUE, add_layer_btn)
        screen.blit(ui_controls.small_font.render("Add Layer", True, ui_controls.BLACK), (add_layer_btn.x + 10, add_layer_btn.y + 5))
        
        rem_layer_btn = pygame.Rect(right_x2, top_y + btn_h + spacing, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.GRAY if is_rnn else ui_controls.LIGHT_BLUE, rem_layer_btn)
        screen.blit(ui_controls.small_font.render("Remove Layer", True, ui_controls.BLACK), (rem_layer_btn.x + 10, rem_layer_btn.y + 5))

        # --- Network Action Controls ---
        switch_net_btn = pygame.Rect(right_x1, top_y + 2*(btn_h + spacing), btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.PURPLE, switch_net_btn)
        screen.blit(ui_controls.small_font.render("Switch Network", True, ui_controls.WHITE), (switch_net_btn.x + 10, switch_net_btn.y + 5))

        rand_btn = pygame.Rect(right_x2, top_y + 2*(btn_h + spacing), btn_w, btn_h)
        clear_btn = pygame.Rect(right_x2, top_y + 3*(btn_h + spacing), btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.ORANGE, rand_btn)
        pygame.draw.rect(screen, ui_controls.GRAY, clear_btn)
        screen.blit(ui_controls.small_font.render("Randomize", True, ui_controls.BLACK), (rand_btn.x + 10, rand_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Clear Network", True, ui_controls.BLACK), (clear_btn.x + 10, clear_btn.y + 5))

        # --- Simulation Controls ---
        sim_y = WINDOW_HEIGHT - 60
        start_btn = pygame.Rect(20, sim_y, btn_w, btn_h)
        pause_btn = pygame.Rect(20 + btn_w + spacing, sim_y, btn_w, btn_h)
        stop_btn = pygame.Rect(20 + 2*(btn_w + spacing), sim_y, btn_w, btn_h)
        auto_train_btn = pygame.Rect(20 + 3*(btn_w + spacing), sim_y, btn_w, btn_h)
        next_cycle_btn = pygame.Rect(20 + 4*(btn_w + spacing), sim_y, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.GREEN, start_btn)
        pygame.draw.rect(screen, ui_controls.YELLOW, pause_btn)
        pygame.draw.rect(screen, ui_controls.RED, stop_btn)
        pygame.draw.rect(screen, ui_controls.CYAN if self.auto_train else ui_controls.GRAY, auto_train_btn)
        pygame.draw.rect(screen, ui_controls.ORANGE, next_cycle_btn)
        screen.blit(ui_controls.small_font.render("Start", True, ui_controls.BLACK), (start_btn.x + 10, start_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Pause", True, ui_controls.BLACK), (pause_btn.x + 10, pause_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Stop", True, ui_controls.BLACK), (stop_btn.x + 10, stop_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Auto Train", True, ui_controls.BLACK), (auto_train_btn.x + 5, auto_train_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Next Cycle", True, ui_controls.BLACK), (next_cycle_btn.x + 5, next_cycle_btn.y + 5))

        # --- Training Data & I/O ---
        td_x = WINDOW_WIDTH - 320
        td_y = WINDOW_HEIGHT - 120
        td_w = 260
        td_h = 90
        pygame.draw.rect(screen, ui_controls.CYAN, (td_x, td_y, td_w, td_h), 2)
        screen.blit(ui_controls.small_font.render("Training Data Interface", True, ui_controls.CYAN), (td_x + 10, td_y + 5))
        load_file_btn = pygame.Rect(td_x + td_w - 140, td_y + 10, 120, 28)
        pygame.draw.rect(screen, ui_controls.GREEN, load_file_btn)
        screen.blit(ui_controls.small_font.render("Load Data File", True, ui_controls.BLACK), (load_file_btn.x + 10, load_file_btn.y + 5))
        save_net_btn = pygame.Rect(td_x + 10, td_y + td_h - 35, 110, 28)
        load_net_btn = pygame.Rect(td_x + 130, td_y + td_h - 35, 110, 28)
        pygame.draw.rect(screen, ui_controls.ORANGE, save_net_btn)
        pygame.draw.rect(screen, ui_controls.CYAN, load_net_btn)
        screen.blit(ui_controls.small_font.render("Save Network", True, ui_controls.BLACK), (save_net_btn.x + 10, save_net_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Load Network", True, ui_controls.BLACK), (load_net_btn.x + 10, load_net_btn.y + 5))
        
        net = self.current_network()
        if hasattr(net, 'training_data') and net.training_data:
            current_input, desired_output = net.training_data[net.current_pattern]
            screen.blit(ui_controls.small_font.render(f"Input: {current_input}", True, ui_controls.WHITE), (WINDOW_WIDTH - 310, WINDOW_HEIGHT - 90))
            screen.blit(ui_controls.small_font.render(f"Output: {desired_output}", True, ui_controls.WHITE), (WINDOW_WIDTH - 310, WINDOW_HEIGHT - 70))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self.running = False
                elif event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_r: self.current_network().reset_network()
                elif event.key == pygame.K_t: self.current_network().train_network()
                elif event.key == pygame.K_a: self.auto_train = not self.auto_train
                elif event.key == pygame.K_n: self.next_pattern()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                net = self.current_network()
                is_rnn = isinstance(net, RecurrentNetwork)

                # --- Training Data & Save/Load Buttons ---
                td_x, td_y, td_w, td_h = WINDOW_WIDTH - 320, WINDOW_HEIGHT - 120, 260, 90
                load_file_btn = pygame.Rect(td_x + td_w - 140, td_y + 10, 120, 28)
                save_net_btn = pygame.Rect(td_x + 10, td_y + td_h - 35, 110, 28)
                load_net_btn = pygame.Rect(td_x + 130, td_y + td_h - 35, 110, 28)
                if load_file_btn.collidepoint(mouse_pos): self._open_training_data_file()
                if save_net_btn.collidepoint(mouse_pos): self._save_network()
                if load_net_btn.collidepoint(mouse_pos): self._load_network()
                
                # --- Speed Controls ---
                if self.speed_slider.rect.collidepoint(mouse_pos): self.speed_slider.dragging = True
                self.max_speed_checkbox.handle_event(event)

                # --- Network Modification Buttons (Right Panel) ---
                right_x1, right_x2 = WINDOW_WIDTH - 260, WINDOW_WIDTH - 130
                btn_w, btn_h, spacing, top_y = 120, 30, 10, 60
                add_neuron_btn = pygame.Rect(right_x1, top_y, btn_w, btn_h)
                rem_neuron_btn = pygame.Rect(right_x1, top_y + btn_h + spacing, btn_w, btn_h)
                add_layer_btn = pygame.Rect(right_x2, top_y, btn_w, btn_h)
                rem_layer_btn = pygame.Rect(right_x2, top_y + btn_h + spacing, btn_w, btn_h)
                switch_net_btn = pygame.Rect(right_x1, top_y + 2 * (btn_h + spacing), btn_w, btn_h)
                rand_btn = pygame.Rect(right_x2, top_y + 2 * (btn_h + spacing), btn_w, btn_h)
                clear_btn = pygame.Rect(right_x2, top_y + 3 * (btn_h + spacing), btn_w, btn_h)

                if switch_net_btn.collidepoint(mouse_pos):
                    self.network_index = (self.network_index + 1) % len(self.network_types)
                    self.current_network().reset_network()
                if add_neuron_btn.collidepoint(mouse_pos):
                    if is_rnn:
                        net.hidden_size += 1
                    else:
                        sel = self.selected_layer - 1
                        if 0 <= sel < len(net.hidden_sizes): net.hidden_sizes[sel] += 1
                    net._init_weights()
                if rem_neuron_btn.collidepoint(mouse_pos):
                    if is_rnn:
                        if net.hidden_size > 1: net.hidden_size -= 1
                    else:
                        sel = self.selected_layer - 1
                        if 0 <= sel < len(net.hidden_sizes) and net.hidden_sizes[sel] > 1: net.hidden_sizes[sel] -= 1
                    net._init_weights()
                if add_layer_btn.collidepoint(mouse_pos) and not is_rnn:
                    net.hidden_sizes.insert(self.selected_layer - 1 if self.selected_layer > 0 else 0, 4)
                    self.selected_layer = min(self.selected_layer + 1, len(net.hidden_sizes) + 1)
                    net._init_weights()
                if rem_layer_btn.collidepoint(mouse_pos) and not is_rnn:
                    if len(net.hidden_sizes) > 0:
                        idx = self.selected_layer - 1
                        if 0 <= idx < len(net.hidden_sizes):
                            net.hidden_sizes.pop(idx)
                            self.selected_layer = max(1, min(self.selected_layer, len(net.hidden_sizes)))
                            net._init_weights()
                if rand_btn.collidepoint(mouse_pos): net._init_weights()
                if clear_btn.collidepoint(mouse_pos):
                    if is_rnn: self.recurrent = RecurrentNetwork(input_size=net.input_size, hidden_size=6, output_size=net.output_size)
                    else: self.feedforward = FeedforwardNetwork(input_size=net.input_size, hidden_sizes=[6], output_size=net.output_size)

                # --- Layer Selection (Feedforward Only) ---
                if not is_rnn:
                    num_layers = 1 + len(net.hidden_sizes) + 1
                    layer_xs = np.linspace(200, 900, num_layers)
                    for l in range(1, num_layers - 1):
                        x = int(layer_xs[l])
                        y_top = 200 - 30
                        y_bottom = 200 + (net.hidden_sizes[l-1] - 1) * 80 + 30
                        if pygame.Rect(x - 40, y_top, 80, y_bottom - y_top).collidepoint(mouse_pos):
                            self.selected_layer = l

                # --- Simulation Control Buttons (Bottom Panel) ---
                sim_y = WINDOW_HEIGHT - 60
                start_btn = pygame.Rect(20, sim_y, btn_w, btn_h)
                pause_btn = pygame.Rect(20 + btn_w + spacing, sim_y, btn_w, btn_h)
                stop_btn = pygame.Rect(20 + 2 * (btn_w + spacing), sim_y, btn_w, btn_h)
                auto_train_btn = pygame.Rect(20 + 3 * (btn_w + spacing), sim_y, btn_w, btn_h)
                next_cycle_btn = pygame.Rect(20 + 4 * (btn_w + spacing), sim_y, btn_w, btn_h)
                if start_btn.collidepoint(mouse_pos): self.paused = False
                if pause_btn.collidepoint(mouse_pos): self.paused = not self.paused
                if stop_btn.collidepoint(mouse_pos): self.paused = True
                if auto_train_btn.collidepoint(mouse_pos): self.auto_train = not self.auto_train
                if next_cycle_btn.collidepoint(mouse_pos):
                    net.train_network()
                    self.next_pattern()

            elif event.type == pygame.MOUSEBUTTONUP:
                self.speed_slider.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.speed_slider.dragging:
                    rel_x = event.pos[0] - self.speed_slider.x
                    rel_x = max(0, min(rel_x, self.speed_slider.width - 1))
                    ratio = rel_x / (self.speed_slider.width - 1)
                    self.speed_slider.value = self.speed_slider.min_val + ratio * (self.speed_slider.max_val - self.speed_slider.min_val)
                    self.simulation_speed = self.speed_slider.value

    def _save_network(self):
        """Saves the current network's state to a .pkl file."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(title="Save Network", defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()

        if not file_path: return

        net = self.current_network()
        is_rnn = isinstance(net, RecurrentNetwork)
        
        data = {'network_type': 'Recurrent' if is_rnn else 'Feedforward'}
        
        if is_rnn:
            data.update({
                'input_size': net.input_size, 'hidden_size': net.hidden_size, 'output_size': net.output_size,
                'learning_rate': net.learning_rate, 'training_data': net.training_data,
                'W_ih': net.W_ih, 'W_hh': net.W_hh, 'W_ho': net.W_ho,
                'b_h': net.b_h, 'b_o': net.b_o
            })
        else: # Feedforward
            data.update({
                'input_size': net.input_size, 'hidden_sizes': net.hidden_sizes, 'output_size': net.output_size,
                'learning_rate': net.learning_rate, 'training_data': net.training_data,
                'weights': net.weights, 'biases': net.biases
            })
            
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Network saved to {file_path}")
        except Exception as e:
            print(f"Failed to save network: {e}")

    def _load_network(self):
        """Loads a network state from a .pkl file, detecting its type."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Load Network", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()

        if not file_path: return
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            network_type = data.get('network_type')
            if network_type == 'Recurrent':
                net = RecurrentNetwork(input_size=data['input_size'], hidden_size=data['hidden_size'], output_size=data['output_size'], learning_rate=data['learning_rate'])
                net.W_ih, net.W_hh, net.W_ho = data['W_ih'], data['W_hh'], data['W_ho']
                net.b_h, net.b_o = data['b_h'], data['b_o']
                self.recurrent = net
                self.network_index = self.network_types.index('Recurrent')
            elif network_type == 'Feedforward':
                net = FeedforwardNetwork(input_size=data['input_size'], hidden_sizes=data['hidden_sizes'], output_size=data['output_size'], learning_rate=data['learning_rate'])
                net.weights, net.biases = data['weights'], data['biases']
                self.feedforward = net
                self.network_index = self.network_types.index('Feedforward')
            else:
                raise ValueError("Unknown network type in file.")
            
            net.training_data = data.get('training_data', [])
            net.current_pattern = 0
            net.reset_network()
            self.next_pattern()
            print(f"{network_type} network loaded from {file_path}")

        except Exception as e:
            print(f"Failed to load network: {e}")

    def _open_training_data_file(self):
        """Opens a .txt file and re-initializes the current network with the new data."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select Training Data File", filetypes=[("Text Files", "*.txt")])
        root.destroy()
        if not file_path: return

        import re, ast
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            patterns = []
            for line in lines:
                matches = re.findall(r'(\[.*?\])', line)
                if len(matches) == 2:
                    inp = ast.literal_eval(matches[0])
                    out = ast.literal_eval(matches[1])
                    if isinstance(inp, list) and isinstance(out, list):
                        patterns.append((inp, out))
            
            if patterns:
                input_size = len(patterns[0][0])
                output_size = len(patterns[0][1])

                # Re-initialize the correct network type
                net_type = self.network_types[self.network_index]
                if net_type == 'Recurrent':
                    self.recurrent = RecurrentNetwork(input_size=input_size, hidden_size=6, output_size=output_size)
                    self.recurrent.training_data = patterns
                    self.recurrent.reset_network()
                else: # Feedforward
                    self.feedforward = FeedforwardNetwork(input_size=input_size, hidden_sizes=[6], output_size=output_size)
                    self.feedforward.training_data = patterns
                    self.feedforward.reset_network()
                
                self.next_pattern()
                print(f"Loaded {len(patterns)} patterns for the {net_type} network.")

        except Exception as e:
            print(f"Failed to load training data file: {e}")

    def next_pattern(self):
        net = self.current_network()
        if not hasattr(net, 'training_data') or not net.training_data: return
        net.current_pattern = (net.current_pattern + 1) % len(net.training_data)
        net.reset_network()
        current_input, _ = net.training_data[net.current_pattern]
        net.set_input_pattern(current_input)

    def update(self):
        if self.max_speed_checkbox.checked:
            self.simulation_speed = 9999.0
        else:
            self.simulation_speed = self.speed_slider.value
        
        if not self.paused:
            net = self.current_network()
            if net.cycle_count < net.max_cycles:
                net.update_network()
            elif self.auto_train:
                net.train_network()
                self.next_pattern()

    def draw(self):
        screen.fill(ui_controls.BLACK)
        net = self.current_network()
        
        selected = self.selected_layer if isinstance(net, FeedforwardNetwork) else None
        net.draw(screen, selected_layer=selected)

        self.draw_modern_controls()
        
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = ui_controls.RED if self.paused else ui_controls.GREEN
        screen.blit(ui_controls.font.render(status, True, status_color), (WINDOW_WIDTH - 300, 20))
        
        self.draw_speed_control()
        pygame.display.flip()

    def draw_speed_control(self):
        speed_x = 20
        speed_y = WINDOW_HEIGHT - 120
        title = ui_controls.small_font.render("Simulation Speed:", True, ui_controls.YELLOW)
        screen.blit(title, (speed_x, speed_y))
        self.speed_slider.draw(screen)
        self.max_speed_checkbox.draw(screen)
        speed_desc = "Ludicrous SPEED!" if self.max_speed_checkbox.checked else self.get_speed_description()
        desc_text = ui_controls.small_font.render(speed_desc, True, ui_controls.WHITE)
        screen.blit(desc_text, (speed_x, speed_y + 30))

    def get_speed_description(self):
        speed = self.simulation_speed
        if speed <= 1.0: return "Very Slow"
        elif speed <= 3.0: return "Slow"
        elif speed <= 7.0: return "Normal"
        elif speed <= 12.0: return "Fast"
        else: return "Very Fast"

    def run(self):
        net = self.current_network()
        if hasattr(net, 'training_data') and net.training_data:
            current_input, _ = net.training_data[net.current_pattern]
            net.set_input_pattern(current_input)
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.simulation_speed)
        
        pygame.quit()
        sys.exit()