import pygame
import sys
import numpy as np
from network import FeedforwardNetwork
import ui_controls
from ui_controls import Slider, Checkbox

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Neurosis")

class Simulation:
    def __init__(self):
        self.network_types = ['Feedforward']
        self.network_type = 0
        self.feedforward = FeedforwardNetwork(input_size=3, hidden_sizes=[6], output_size=2, learning_rate=0.05)
        self.selected_layer = 1 if len(self.feedforward.hidden_sizes) > 0 else 0
        self.running = True
        self.paused = True
        self.auto_train = True
        self.clock = pygame.time.Clock()
        self.speed_slider = Slider(20, WINDOW_HEIGHT - 100, 200, 10, 0.5, 20.0, 5.0)
        self.simulation_speed = 5.0
        self.max_speed_checkbox = Checkbox(250, WINDOW_HEIGHT - 100, 15, "Max Speed", False)
    def draw_modern_controls(self):
        right_x1 = WINDOW_WIDTH - 260
        right_x2 = WINDOW_WIDTH - 130
        btn_w = 120
        btn_h = 30
        spacing = 10
        top_y = 60
        add_neuron_btn = pygame.Rect(right_x1, top_y, btn_w, btn_h)
        rem_neuron_btn = pygame.Rect(right_x1, top_y + btn_h + spacing, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, add_neuron_btn)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, rem_neuron_btn)
        text = ui_controls.small_font.render("Add Neuron", True, ui_controls.BLACK)
        screen.blit(text, (add_neuron_btn.x + 10, add_neuron_btn.y + 5))
        text = ui_controls.small_font.render("Remove Neuron", True, ui_controls.BLACK)
        screen.blit(text, (rem_neuron_btn.x + 10, rem_neuron_btn.y + 5))
        add_layer_btn = pygame.Rect(right_x2, top_y, btn_w, btn_h)
        rem_layer_btn = pygame.Rect(right_x2, top_y + btn_h + spacing, btn_w, btn_h)
        rand_btn = pygame.Rect(right_x2, top_y + 2*(btn_h + spacing), btn_w, btn_h)
        clear_btn = pygame.Rect(right_x2, top_y + 3*(btn_h + spacing), btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, add_layer_btn)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, rem_layer_btn)
        pygame.draw.rect(screen, ui_controls.ORANGE, rand_btn)
        pygame.draw.rect(screen, ui_controls.GRAY, clear_btn)
        text = ui_controls.small_font.render("Add Layer", True, ui_controls.BLACK)
        screen.blit(text, (add_layer_btn.x + 10, add_layer_btn.y + 5))
        text = ui_controls.small_font.render("Remove Layer", True, ui_controls.BLACK)
        screen.blit(text, (rem_layer_btn.x + 10, rem_layer_btn.y + 5))
        text = ui_controls.small_font.render("Randomize", True, ui_controls.BLACK)
        screen.blit(text, (rand_btn.x + 10, rand_btn.y + 5))
        text = ui_controls.small_font.render("Clear Network", True, ui_controls.BLACK)
        screen.blit(text, (clear_btn.x + 10, clear_btn.y + 5))
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
        text = ui_controls.small_font.render("Start", True, ui_controls.BLACK)
        screen.blit(text, (start_btn.x + 10, start_btn.y + 5))
        text = ui_controls.small_font.render("Pause", True, ui_controls.BLACK)
        screen.blit(text, (pause_btn.x + 10, pause_btn.y + 5))
        text = ui_controls.small_font.render("Stop", True, ui_controls.BLACK)
        screen.blit(text, (stop_btn.x + 10, stop_btn.y + 5))
        text = ui_controls.small_font.render("Auto Train", True, ui_controls.BLACK)
        screen.blit(text, (auto_train_btn.x + 5, auto_train_btn.y + 5))
        text = ui_controls.small_font.render("Next Cycle", True, ui_controls.BLACK)
        screen.blit(text, (next_cycle_btn.x + 5, next_cycle_btn.y + 5))
        td_x = WINDOW_WIDTH - 320
        td_y = WINDOW_HEIGHT - 120
        td_w = 260
        td_h = 90
        pygame.draw.rect(screen, ui_controls.CYAN, (td_x, td_y, td_w, td_h), 2)
        text = ui_controls.small_font.render("Training Data Interface", True, ui_controls.CYAN)
        screen.blit(text, (td_x + 10, td_y + 5))
        load_file_btn = pygame.Rect(td_x + td_w - 140, td_y + 10, 120, 28)
        pygame.draw.rect(screen, ui_controls.GREEN, load_file_btn)
        text = ui_controls.small_font.render("Load Data File", True, ui_controls.BLACK)
        screen.blit(text, (load_file_btn.x + 10, load_file_btn.y + 5))
        save_net_btn = pygame.Rect(td_x + 10, td_y + td_h - 35, 110, 28)
        load_net_btn = pygame.Rect(td_x + 130, td_y + td_h - 35, 110, 28)
        pygame.draw.rect(screen, ui_controls.ORANGE, save_net_btn)
        pygame.draw.rect(screen, ui_controls.CYAN, load_net_btn)
        text = ui_controls.small_font.render("Save Network", True, ui_controls.BLACK)
        screen.blit(text, (save_net_btn.x + 10, save_net_btn.y + 5))
        text = ui_controls.small_font.render("Load Network", True, ui_controls.BLACK)
        screen.blit(text, (load_net_btn.x + 10, load_net_btn.y + 5))
        net = self.current_network()
        if hasattr(net, 'training_data') and net.training_data:
            current_input, desired_output = net.training_data[net.current_pattern]
            input_text = f"Input: {current_input}"
            output_text = f"Output: {desired_output}"
            text1 = ui_controls.small_font.render(input_text, True, ui_controls.WHITE)
            text2 = ui_controls.small_font.render(output_text, True, ui_controls.WHITE)
            screen.blit(text1, (WINDOW_WIDTH - 310, WINDOW_HEIGHT - 90))
            screen.blit(text2, (WINDOW_WIDTH - 310, WINDOW_HEIGHT - 70))
        edit_text = "Edit patterns in code or add UI for custom input."
        text = ui_controls.tiny_font.render(edit_text, True, ui_controls.GRAY)
        screen.blit(text, (WINDOW_WIDTH - 310, WINDOW_HEIGHT - 50))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.feedforward.reset_network()
                elif event.key == pygame.K_t:
                    self.feedforward.train_network()
                elif event.key == pygame.K_a:
                    self.auto_train = not self.auto_train
                elif event.key == pygame.K_n:
                    self.next_pattern()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                td_x = WINDOW_WIDTH - 320
                td_y = WINDOW_HEIGHT - 120
                td_w = 260
                td_h = 90
                load_file_btn = pygame.Rect(td_x + td_w - 140, td_y + 10, 120, 28)
                save_net_btn = pygame.Rect(td_x + 10, td_y + td_h - 35, 110, 28)
                load_net_btn = pygame.Rect(td_x + 130, td_y + td_h - 35, 110, 28)
                if load_file_btn.collidepoint(mouse_pos):
                    self._open_training_data_file()
                if save_net_btn.collidepoint(mouse_pos):
                    self._save_network()
                if load_net_btn.collidepoint(mouse_pos):
                    self._load_network()
                if self.speed_slider.rect.collidepoint(mouse_pos):
                    self.speed_slider.dragging = True
                self.max_speed_checkbox.handle_event(event)
                right_x1 = WINDOW_WIDTH - 260
                right_x2 = WINDOW_WIDTH - 130
                btn_w = 120
                btn_h = 30
                spacing = 10
                top_y = 60
                add_neuron_btn = pygame.Rect(right_x1, top_y, btn_w, btn_h)
                rem_neuron_btn = pygame.Rect(right_x1, top_y + btn_h + spacing, btn_w, btn_h)
                add_layer_btn = pygame.Rect(right_x2, top_y, btn_w, btn_h)
                rem_layer_btn = pygame.Rect(right_x2, top_y + btn_h + spacing, btn_w, btn_h)
                rand_btn = pygame.Rect(right_x2, top_y + 2*(btn_h + spacing), btn_w, btn_h)
                clear_btn = pygame.Rect(right_x2, top_y + 3*(btn_h + spacing), btn_w, btn_h)
                num_layers = 1 + len(self.feedforward.hidden_sizes) + 1
                layer_xs = np.linspace(200, 900, num_layers)
                layer_sizes = [self.feedforward.input_size] + self.feedforward.hidden_sizes + [self.feedforward.output_size]
                for l in range(1, num_layers-1):
                    x = int(layer_xs[l])
                    y_top = 200 - 30
                    y_bottom = 200 + (layer_sizes[l]-1)*80 + 30
                    rect = pygame.Rect(x-40, y_top, 80, y_bottom-y_top)
                    if rect.collidepoint(mouse_pos):
                        self.selected_layer = l
                if add_neuron_btn.collidepoint(mouse_pos):
                    sel = self.selected_layer-1
                    if 0 <= sel < len(self.feedforward.hidden_sizes):
                        self.feedforward.hidden_sizes[sel] += 1
                        self.feedforward._init_weights()
                if rem_neuron_btn.collidepoint(mouse_pos):
                    sel = self.selected_layer-1
                    if 0 <= sel < len(self.feedforward.hidden_sizes) and self.feedforward.hidden_sizes[sel] > 1:
                        self.feedforward.hidden_sizes[sel] -= 1
                        self.feedforward._init_weights()
                if add_layer_btn.collidepoint(mouse_pos):
                    self.feedforward.hidden_sizes.insert(self.selected_layer-1 if self.selected_layer > 0 else 0, 4)
                    self.selected_layer += 1
                    self.feedforward._init_weights()
                if rem_layer_btn.collidepoint(mouse_pos):
                    if len(self.feedforward.hidden_sizes) > 1:
                        idx = self.selected_layer-1
                        if 0 <= idx < len(self.feedforward.hidden_sizes):
                            self.feedforward.hidden_sizes.pop(idx)
                            self.selected_layer = max(1, min(self.selected_layer, len(self.feedforward.hidden_sizes)))
                            self.feedforward._init_weights()
                if rand_btn.collidepoint(mouse_pos):
                    self.feedforward._init_weights()
                if clear_btn.collidepoint(mouse_pos):
                    self.feedforward = FeedforwardNetwork(input_size=self.feedforward.input_size, hidden_sizes=[6], output_size=self.feedforward.output_size, learning_rate=self.feedforward.learning_rate)
                    self.selected_layer = 1
                sim_y = WINDOW_HEIGHT - 60
                start_btn = pygame.Rect(20, sim_y, btn_w, btn_h)
                pause_btn = pygame.Rect(20 + btn_w + spacing, sim_y, btn_w, btn_h)
                stop_btn = pygame.Rect(20 + 2*(btn_w + spacing), sim_y, btn_w, btn_h)
                auto_train_btn = pygame.Rect(20 + 3*(btn_w + spacing), sim_y, btn_w, btn_h)
                next_cycle_btn = pygame.Rect(20 + 4*(btn_w + spacing), sim_y, btn_w, btn_h)
                if start_btn.collidepoint(mouse_pos):
                    self.paused = False
                if pause_btn.collidepoint(mouse_pos):
                    self.paused = not self.paused
                if stop_btn.collidepoint(mouse_pos):
                    self.paused = True
                if auto_train_btn.collidepoint(mouse_pos):
                    self.auto_train = not self.auto_train
                if next_cycle_btn.collidepoint(mouse_pos):
                    net = self.current_network()
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
        import tkinter as tk
        from tkinter import filedialog
        import pickle
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(title="Save Network", defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()
        if file_path:
            net = self.feedforward
            data = {
                'input_size': net.input_size,
                'hidden_sizes': net.hidden_sizes,
                'output_size': net.output_size,
                'learning_rate': net.learning_rate,
                'weights': net.weights,
                'biases': net.biases,
                'training_data': net.training_data,
                'current_pattern': net.current_pattern
            }
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Network saved to {file_path}")
            except Exception as e:
                print(f"Failed to save network: {e}")

    def _load_network(self):
        import tkinter as tk
        from tkinter import filedialog
        import pickle
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Load Network", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                net = FeedforwardNetwork(
                    input_size=data['input_size'],
                    hidden_sizes=data['hidden_sizes'],
                    output_size=data['output_size'],
                    learning_rate=data['learning_rate']
                )
                net.weights = data['weights']
                net.biases = data['biases']
                net.training_data = data.get('training_data', [])
                net.current_pattern = data.get('current_pattern', 0)
                net.reset_network()
                self.feedforward = net
                self.selected_layer = 1 if len(self.feedforward.hidden_sizes) > 0 else 0
                print(f"Network loaded from {file_path}")
            except Exception as e:
                print(f"Failed to load network: {e}")

    def _open_training_data_file(self):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select Training Data File", filetypes=[("Text Files", "*.txt")])
        root.destroy()
        if file_path:
            import re
            import ast
            try:
                with open(file_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                patterns = []
                for line in lines:
                    try:
                        matches = re.findall(r'(\[.*?\])', line)
                        if len(matches) == 2:
                            inp = ast.literal_eval(matches[0])
                            out = ast.literal_eval(matches[1])
                            if isinstance(inp, list) and isinstance(out, list):
                                patterns.append((inp, out))
                            else:
                                print(f"Error: Parsed values are not lists in line: {line}")
                        else:
                            print(f"Error: Could not find two lists in line: {line}")
                    except Exception as e:
                        print(f"Error parsing line: {line} ({e})")
                if patterns:
                    input_size = len(patterns[0][0])
                    output_size = len(patterns[0][1])
                    self.feedforward = FeedforwardNetwork(input_size=input_size, hidden_sizes=[6], output_size=output_size, learning_rate=0.05)
                    self.feedforward.training_data = patterns
                    self.feedforward.current_pattern = 0
                    self.feedforward.reset_network()
                    self.selected_layer = 1 if len(self.feedforward.hidden_sizes) > 0 else 0
            except Exception as e:
                print(f"Failed to load training data file: {e}")

    def current_network(self):
        return self.feedforward

    def next_pattern(self):
        net = self.current_network()
        net.current_pattern = (net.current_pattern + 1) % len(net.training_data)
        net.reset_network()
        current_input, _ = net.training_data[net.current_pattern]
        net.set_input_pattern(current_input)

    def update(self):
        # Update simulation speed from slider or max speed checkbox
        if self.max_speed_checkbox.checked:
            self.simulation_speed = 9999.0 # Ludicrous speed!!
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
        self.feedforward.draw(screen, selected_layer=self.selected_layer)
        self.draw_modern_controls()
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = ui_controls.RED if self.paused else ui_controls.GREEN
        text = ui_controls.font.render(status, True, status_color)
        screen.blit(text, (WINDOW_WIDTH - 300, 20))
        self.draw_speed_control()
        pygame.display.flip()

    def draw_speed_control(self):
        speed_x = 20
        speed_y = WINDOW_HEIGHT - 120
        title = ui_controls.small_font.render("Simulation Speed:", True, ui_controls.YELLOW)
        screen.blit(title, (speed_x, speed_y))
        self.speed_slider.draw(screen)
        self.max_speed_checkbox.draw(screen)
        speed_desc = "Ludicrous SPEED - Rapid simulation" if self.max_speed_checkbox.checked else self.get_speed_description()
        desc_text = ui_controls.small_font.render(speed_desc, True, ui_controls.WHITE)
        screen.blit(desc_text, (speed_x, speed_y + 30))

    def get_speed_description(self):
        speed = self.simulation_speed
        if speed <= 1.0:
            return "Very Slow - Step by step observation"
        elif speed <= 3.0:
            return "Slow - Detailed observation"
        elif speed <= 7.0:
            return "Normal - Balanced speed"
        elif speed <= 12.0:
            return "Fast - Quick overview"
        else:
            return "Very Fast - Rapid simulation"

    def run(self):
        net = self.current_network()
        if net.training_data:
            current_input, _ = net.training_data[net.current_pattern]
            net.set_input_pattern(current_input)
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.simulation_speed)
        pygame.quit()
        sys.exit()
