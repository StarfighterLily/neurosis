import pygame
import sys
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from network_player import FeedforwardNetworkRunOnly
import ui_controls

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Neurosis Network Inference")

class Simulation:
    def __init__(self):
        self.feedforward = FeedforwardNetworkRunOnly(input_size=3, hidden_sizes=[6], output_size=2)
        self.selected_layer = 1
        self.running = True
        self.clock = pygame.time.Clock()

    def draw_controls(self):
        """Draws UI controls for network modification and saving/loading."""
        right_x1 = WINDOW_WIDTH - 280
        right_x2 = WINDOW_WIDTH - 140
        btn_w = 120
        btn_h = 30
        spacing = 10
        top_y = 60

        # --- Neuron/Layer Controls ---
        add_neuron_btn = pygame.Rect(right_x1, top_y, btn_w, btn_h)
        rem_neuron_btn = pygame.Rect(right_x1, top_y + btn_h + spacing, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, add_neuron_btn)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, rem_neuron_btn)
        screen.blit(ui_controls.small_font.render("Add Neuron", True, ui_controls.BLACK), (add_neuron_btn.x + 10, add_neuron_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Remove Neuron", True, ui_controls.BLACK), (rem_neuron_btn.x + 10, rem_neuron_btn.y + 5))

        add_layer_btn = pygame.Rect(right_x2, top_y, btn_w, btn_h)
        rem_layer_btn = pygame.Rect(right_x2, top_y + btn_h + spacing, btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, add_layer_btn)
        pygame.draw.rect(screen, ui_controls.LIGHT_BLUE, rem_layer_btn)
        screen.blit(ui_controls.small_font.render("Add Layer", True, ui_controls.BLACK), (add_layer_btn.x + 10, add_layer_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Remove Layer", True, ui_controls.BLACK), (rem_layer_btn.x + 10, rem_layer_btn.y + 5))

        # --- Network Action Controls ---
        rand_btn = pygame.Rect(right_x1, top_y + 3 * (btn_h + spacing), btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.ORANGE, rand_btn)
        screen.blit(ui_controls.small_font.render("Randomize Weights", True, ui_controls.BLACK), (rand_btn.x + 5, rand_btn.y + 5))
        
        save_net_btn = pygame.Rect(right_x2, top_y + 3 * (btn_h + spacing), btn_w, btn_h)
        load_net_btn = pygame.Rect(right_x2, top_y + 4 * (btn_h + spacing), btn_w, btn_h)
        pygame.draw.rect(screen, ui_controls.GREEN, save_net_btn)
        pygame.draw.rect(screen, ui_controls.CYAN, load_net_btn)
        screen.blit(ui_controls.small_font.render("Save Network", True, ui_controls.BLACK), (save_net_btn.x + 10, save_net_btn.y + 5))
        screen.blit(ui_controls.small_font.render("Load Network", True, ui_controls.BLACK), (load_net_btn.x + 10, load_net_btn.y + 5))

        # --- Instructions ---
        instr_y = WINDOW_HEIGHT - 80
        screen.blit(ui_controls.font.render("Instructions:", True, ui_controls.YELLOW), (20, instr_y))
        screen.blit(ui_controls.small_font.render("- Click input neurons (left) to toggle state (0/1).", True, ui_controls.WHITE), (20, instr_y + 25))
        screen.blit(ui_controls.small_font.render("- Click a hidden layer to select it for modification.", True, ui_controls.WHITE), (20, instr_y + 45))


    def get_neuron_positions(self, network):
        """Calculates and returns the screen positions of all neurons."""
        positions = []
        num_layers = 1 + len(network.hidden_sizes) + 1
        layer_xs = np.linspace(200, 900, num_layers)
        layer_sizes = [network.input_size] + network.hidden_sizes + [network.output_size]
        
        for l, (x, n) in enumerate(zip(layer_xs, layer_sizes)):
            layer_pos = []
            total_height = (n - 1) * 80
            start_y = (WINDOW_HEIGHT - total_height) / 2 - 100
            for i in range(n):
                y = start_y + i * 80
                layer_pos.append((int(x), y))
            positions.append(layer_pos)
        return positions

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        net = self.feedforward
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # --- Check for clicks on UI buttons ---
                right_x1 = WINDOW_WIDTH - 280
                right_x2 = WINDOW_WIDTH - 140
                btn_w = 120
                btn_h = 30
                spacing = 10
                top_y = 60

                if pygame.Rect(right_x1, top_y, btn_w, btn_h).collidepoint(mouse_pos):
                    sel = self.selected_layer - 1
                    if 0 <= sel < len(net.hidden_sizes):
                        net.hidden_sizes[sel] += 1
                        net.reset_network()
                elif pygame.Rect(right_x1, top_y + btn_h + spacing, btn_w, btn_h).collidepoint(mouse_pos):
                    sel = self.selected_layer - 1
                    if 0 <= sel < len(net.hidden_sizes) and net.hidden_sizes[sel] > 1:
                        net.hidden_sizes[sel] -= 1
                        net.reset_network()
                elif pygame.Rect(right_x2, top_y, btn_w, btn_h).collidepoint(mouse_pos):
                    net.hidden_sizes.insert(self.selected_layer-1 if self.selected_layer > 0 else 0, 4)
                    self.selected_layer = min(self.selected_layer + 1, len(net.hidden_sizes))
                    net.reset_network()
                elif pygame.Rect(right_x2, top_y + btn_h + spacing, btn_w, btn_h).collidepoint(mouse_pos):
                     if len(net.hidden_sizes) > 0:
                        idx = self.selected_layer - 1
                        if 0 <= idx < len(net.hidden_sizes):
                            net.hidden_sizes.pop(idx)
                            self.selected_layer = max(1, min(self.selected_layer, len(net.hidden_sizes)))
                            net.reset_network()
                elif pygame.Rect(right_x1, top_y + 3*(btn_h + spacing), btn_w, btn_h).collidepoint(mouse_pos):
                    net.reset_network()
                elif pygame.Rect(right_x2, top_y + 3*(btn_h + spacing), btn_w, btn_h).collidepoint(mouse_pos):
                    self._save_network()
                elif pygame.Rect(right_x2, top_y + 4*(btn_h + spacing), btn_w, btn_h).collidepoint(mouse_pos):
                    self._load_network()

                # --- Check for clicks on neurons ---
                neuron_positions = self.get_neuron_positions(net)
                # 1. Input neurons
                for i, pos in enumerate(neuron_positions[0]):
                    if pygame.Rect(pos[0]-20, pos[1]-20, 40, 40).collidepoint(mouse_pos):
                        net.last_input[i, 0] = 1 - net.last_input[i, 0] # Toggle 0/1
                        net.forward(net.last_input) # Re-run the network
                
                # 2. Hidden layers (for selection)
                for l_idx, layer in enumerate(neuron_positions[1:-1], start=1):
                    for n_idx, pos in enumerate(layer):
                        if pygame.Rect(pos[0]-20, pos[1]-20, 40, 40).collidepoint(mouse_pos):
                            self.selected_layer = l_idx
                            break

    def _save_network(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(title="Save Network", defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()
        if file_path:
            net = self.feedforward
            data = {
                'input_size': net.input_size, 'hidden_sizes': net.hidden_sizes,
                'output_size': net.output_size, 'weights': net.weights, 'biases': net.biases,
            }
            try:
                with open(file_path, 'wb') as f: pickle.dump(data, f)
                print(f"Network saved to {file_path}")
            except Exception as e:
                print(f"Failed to save network: {e}")

    def _load_network(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Load Network", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()
        if file_path:
            try:
                with open(file_path, 'rb') as f: data = pickle.load(f)
                net = FeedforwardNetworkRunOnly(
                    input_size=data['input_size'], hidden_sizes=data['hidden_sizes'],
                    output_size=data['output_size']
                )
                net.weights = data['weights']
                net.biases = data['biases']
                net.forward(net.last_input) # Run forward pass with loaded weights
                self.feedforward = net
                self.selected_layer = 1 if len(self.feedforward.hidden_sizes) > 0 else 0
                print(f"Network loaded from {file_path}")
            except Exception as e:
                print(f"Failed to load network: {e}")

    def draw(self):
        screen.fill(ui_controls.BLACK)
        self.feedforward.draw(screen, selected_layer=self.selected_layer)
        self.draw_controls()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60) # Cap framerate
        pygame.quit()
        sys.exit()