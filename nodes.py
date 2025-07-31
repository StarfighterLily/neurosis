import pygame
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from network_player import FeedforwardNetworkRunOnly
import pyautogui


# --- Colors ---
WHITE = ( 255, 255, 255 )
BLACK = ( 0, 0, 0 )
GREY = ( 150, 150, 150 )
NODE_BODY_COLOR = ( 100, 100, 120 )
NODE_BORDER_COLOR = ( 200, 200, 220 )
CONNECTION_COLOR = ( 200, 200, 100 )
SOCKET_COLOR = ( 50, 150, 250 )
INPUT_BOX_COLOR = ( 30, 30, 40 )

# --- Node Base Class ---
class Node:
    def __init__( self, x, y, width, height, title="Node" ):
        self.rect = pygame.Rect( x, y, width, height )
        self.min_width = 80
        self.min_height = 50
        self.title = title
        self.is_dragging = False
        self.is_resizing = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.id = id( self )

        self.input_sockets = []
        self.output_sockets = []
        self.values = {} # To store computed values for outputs

        # --- Handle for resizing ---
        self.resize_handle_rect = pygame.Rect( self.rect.right - 10, self.rect.bottom - 10, 10, 10)

    def add_input( self, name ):
        self.input_sockets.append( { 'name': name, 'pos': ( 0,0 ), 'rect': None, 'connection': None } )

    def add_output( self, name ):
        self.output_sockets.append( { 'name': name, 'pos': ( 0,0 ), 'rect': None } )
        self.values[ name ] = 0 # Default output value

    def _update_socket_positions( self ):
        # Input sockets on the left
        input_spacing = self.rect.height / ( len( self.input_sockets ) + 1 )
        for i, sock in enumerate( self.input_sockets ):
            sock[ 'pos' ] = ( self.rect.left, self.rect.top + int( input_spacing * ( i + 1 ) ) )
            sock[ 'rect' ] = pygame.Rect( sock[ 'pos' ][ 0 ] - 5, sock[ 'pos' ][ 1 ] - 5, 10, 10 )

        # Output sockets on the right
        output_spacing = self.rect.height / ( len( self.output_sockets ) + 1 )
        for i, sock in enumerate( self.output_sockets ):
            sock[ 'pos' ] = ( self.rect.right, self.rect.top + int( output_spacing * ( i + 1 ) ) )
            sock[ 'rect' ] = pygame.Rect( sock[ 'pos' ][ 0 ] - 5, sock[ 'pos' ][ 1 ] - 5, 10, 10 )

        # Update resize handle position
        self.resize_handle_rect.topleft = ( self.rect.right - 10, self.rect.bottom - 10 )

    def handle_event( self, event, global_state, connections ):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left-click
                # Start resizing
                if self.resize_handle_rect.collidepoint( event.pos ):
                    self.is_resizing = True
                    return True

                # Start a connection from an output socket
                for sock in self.output_sockets:
                    if sock[ 'rect' ].collidepoint( event.pos ):
                        global_state[ 'is_drawing_connection' ] = True
                        global_state[ 'connection_start_node' ] = self
                        global_state[ 'connection_start_socket' ] = sock
                        return True

                # Start dragging the node
                if self.rect.collidepoint( event.pos ):
                    self.is_dragging = True
                    self.drag_offset_x = self.rect.x - event.pos[ 0 ]
                    self.drag_offset_y = self.rect.y - event.pos[ 1 ]
                    return True

            elif event.button == 3: # Right-click
                 # Disconnect an input socket
                 for sock in self.input_sockets:
                    if sock[ 'rect' ].collidepoint( event.pos ) and sock[ 'connection' ] is not None:
                        # Find and remove the connection from the global list
                        for conn in connections[:]:
                            if conn[ 'target_node' ] == self and conn[ 'target_socket' ] == sock:
                                connections.remove( conn )
                                break
                        sock[ 'connection' ] = None # Clear local link
                        return True


        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self.is_dragging:
                    self.is_dragging = False
                    return True
                if self.is_resizing:
                    self.is_resizing = False
                    return True


        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                self.rect.x = event.pos[ 0 ] + self.drag_offset_x
                self.rect.y = event.pos[ 1 ] + self.drag_offset_y
                self._update_socket_positions()
                return True
            if self.is_resizing:
                new_width = event.pos[0] - self.rect.left
                new_height = event.pos[1] - self.rect.top
                self.rect.width = max(self.min_width, new_width)
                self.rect.height = max(self.min_height, new_height)
                self._update_socket_positions()
                return True
        return False

    def draw( self, surface, font ):
        # Draw body
        pygame.draw.rect( surface, NODE_BODY_COLOR, self.rect, border_radius=5 )
        pygame.draw.rect( surface, NODE_BORDER_COLOR, self.rect, 2, border_radius=5 )

        # Draw title
        title_surf = font.render( self.title, True, WHITE )
        title_rect = title_surf.get_rect( center=( self.rect.centerx, self.rect.top + 15 ) )
        surface.blit( title_surf, title_rect )

        # Draw sockets
        for sock in self.input_sockets + self.output_sockets:
            pygame.draw.rect( surface, SOCKET_COLOR, sock[ 'rect' ], border_radius=2 )
            pygame.draw.rect( surface, WHITE, sock[ 'rect' ], 1, border_radius=2 )
        
        # Draw resize handle
        pygame.draw.rect(surface, NODE_BORDER_COLOR, self.resize_handle_rect)

    def compute( self ):
        pass

# --- Specific Node Implementations ---

# --- Neural Network Node ---
class NeuralNetNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y, 150, 80, title="Neural Net")
        self.network = None
        self.network_file = "None loaded"
        self.last_click_time = 0
        # Sockets are created dynamically upon loading a network.

    def load_network(self):
        """Opens a file dialog to load a .pkl network file."""
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        file_path = filedialog.askopenfilename(
            title="Load Neurosis Network",
            filetypes=[("Pickle Files", "*.pkl")]
        )
        root.destroy()

        if not file_path:
            return  # User cancelled the dialog

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Create a run-only network instance from the loaded data
            self.network = FeedforwardNetworkRunOnly(
                input_size=data['input_size'],
                hidden_sizes=data['hidden_sizes'],
                output_size=data['output_size']
            )
            self.network.weights = data['weights']
            self.network.biases = data['biases']

            # Update the node's appearance and properties
            self.network_file = file_path.split('/')[-1]
            self.title = "Neural Net" # Keep title short

            # Clear any previous sockets and create new ones
            self.input_sockets.clear()
            self.output_sockets.clear()
            self.values.clear()

            for i in range(self.network.input_size):
                self.add_input(f"in_{i}")
            for i in range(self.network.output_size):
                self.add_output(f"out_{i}")

            # Adjust node height to comfortably fit all sockets
            self.rect.height = max(self.min_height, 30 + 20 * max(len(self.input_sockets), len(self.output_sockets)))
            self._update_socket_positions()

            print(f"Network '{self.network_file}' loaded successfully.")

        except Exception as e:
            print(f"Failed to load or parse network file: {e}")
            self.network = None
            # Reset to a default state
            self.input_sockets.clear()
            self.output_sockets.clear()
            self._update_socket_positions()


    def handle_event(self, event, global_state, connections):
        # Handle double-click to trigger the load_network method
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # Do not trigger loading if clicking on the resize handle or a socket
                if self.resize_handle_rect.collidepoint(event.pos):
                    return super().handle_event(event, global_state, connections)
                
                is_on_socket = False
                for sock in self.input_sockets + self.output_sockets:
                    if sock['rect'].collidepoint(event.pos):
                        is_on_socket = True
                        break
                if is_on_socket:
                     return super().handle_event(event, global_state, connections)

                # Check for double-click
                current_time = pygame.time.get_ticks()
                if current_time - self.last_click_time < 500:
                    self.load_network()
                    self.is_dragging = False  # Prevent starting a drag on a double-click
                    return True  # Event has been handled
                self.last_click_time = current_time

        # Fallback to base class for standard interactions like dragging, resizing, etc.
        return super().handle_event(event, global_state, connections)

    def compute(self):
        """
        Gathers inputs, runs the neural network's forward pass,
        and places the results on the output sockets.
        """
        if not self.network:
            return

        # 1. Gather inputs from connected sockets
        inputs = np.zeros((self.network.input_size, 1))
        for i, sock in enumerate(self.input_sockets):
            if sock.get('connection'):
                source_node = sock['connection']['source_node']
                source_socket_name = sock['connection']['source_socket']['name']
                
                try:
                    # Ensure the input value is a number
                    value = float(source_node.values.get(source_socket_name, 0))
                    inputs[i, 0] = value
                except (ValueError, TypeError):
                    inputs[i, 0] = 0 # Default to 0 if conversion fails
            else:
                inputs[i, 0] = 0  # Default for unconnected inputs

        # 2. Run the network's forward pass
        output_activations = self.network.forward(inputs)

        # 3. Set the output values for the node's output sockets
        for i, sock in enumerate(self.output_sockets):
            if i < len(output_activations):
                self.values[sock['name']] = output_activations[i, 0]
            else:
                self.values[sock['name']] = 0 # Should not happen with correct setup

    def draw(self, surface, font):
        super().draw(surface, font)
        
        # Display status information on the node
        if self.network:
            status_text = self.network_file
            info_text = f"I/O: {self.network.input_size} / {self.network.output_size}"
        else:
            status_text = "No Network"
            info_text = "Dbl-click to load"

        status_surf = font.render(status_text, True, WHITE)
        info_surf = font.render(info_text, True, (200, 200, 255))
        
        status_rect = status_surf.get_rect(center=(self.rect.centerx, self.rect.centery - 8))
        info_rect = info_surf.get_rect(center=(self.rect.centerx, self.rect.centery + 10))
        
        surface.blit(status_surf, status_rect)
        surface.blit(info_surf, info_rect)


# --- Input nodes ---

class MouseInputNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y, 120, 100, title="Mouse Input")
        self.add_output("x")
        self.add_output("y")
        self.add_output("btn_L")
        self.add_output("btn_M")
        self.add_output("btn_R")
        self._update_socket_positions()

    def compute(self):
        x, y = pygame.mouse.get_pos()
        btn_l, btn_m, btn_r = pygame.mouse.get_pressed()

        self.values["x"] = x
        self.values["y"] = y
        self.values["btn_L"] = 1 if btn_l else 0
        self.values["btn_M"] = 1 if btn_m else 0
        self.values["btn_R"] = 1 if btn_r else 0

class ToggleNode( Node ):
    def __init__( self, x, y, initial_state=False ):
        super().__init__( x, y, 100, 60, title="Toggle" )
        self.value = 1 if initial_state else 0
        self.add_output( "out" )
        self._update_socket_positions()

    def handle_event( self, event, global_state, connections ):
        # Check for a left-click to toggle the state
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint( event.pos ):
                # Ensure the click is not on a socket or the resize handle
                on_socket = any(sock['rect'].collidepoint(event.pos) for sock in self.output_sockets)
                on_resize_handle = self.resize_handle_rect.collidepoint(event.pos)

                if not on_socket and not on_resize_handle:
                    self.value = 1 - self.value # Flips between 0 and 1
                    return True # Event was handled

        # Fallback to the base class for other events (dragging, connecting, etc.)
        return super().handle_event( event, global_state, connections )

    def compute( self ):
        self.values[ "out" ] = self.value

    def draw( self, surface, font ):
        super().draw( surface, font )
        
        # Display the current state on the node
        state_text = "ON" if self.value == 1 else "OFF"
        text_color = (100, 255, 100) if self.value == 1 else (255, 100, 100) # Green for ON, Red for OFF
        
        value_surf = font.render( state_text, True, text_color )
        value_rect = value_surf.get_rect( center=self.rect.center )
        surface.blit( value_surf, value_rect )


class FloatNode( Node ):
    def __init__( self, x, y, value=1.0 ):
        super().__init__( x, y, 100, 60, title="Float" )
        self.value = value
        self.add_output( "out" )
        self._update_socket_positions()
        self.editing = False
        self.input_text = str( self.value )
        self.last_click_time = 0

    def handle_event( self, event, global_state, connections ):
        # --- Handle keyboard input when in edit mode ---
        if self.editing:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        self.value = float( self.input_text )
                    except ValueError:
                        self.value = 0.0 # Default to 0 if input is invalid
                    self.editing = False
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[ :-1 ]
                else:
                    self.input_text += event.unicode
                return True # Event handled

            if event.type == pygame.MOUSEBUTTONDOWN and not self.rect.collidepoint( event.pos ):
                self.editing = False # Click outside to cancel editing
                self.input_text = str( self.value ) # Revert text
                
        # --- Handle mouse clicks for entering edit mode and standard dragging ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint( event.pos ):
                # Prevent editing when resizing
                if self.resize_handle_rect.collidepoint(event.pos):
                    return super().handle_event(event, global_state, connections)
                current_time = pygame.time.get_ticks()
                # Check for double-click (e.g., within 500 milliseconds)
                if current_time - self.last_click_time < 500:
                    self.editing = True
                    self.input_text = str( self.value )
                    self.is_dragging = False # Prevent dragging on double-click
                    return True # Event handled
                self.last_click_time = current_time

        # --- Fallback to base class event handling (for dragging, etc.) ---
        # Ensure editing mode doesn't interfere with starting a drag
        if not self.editing:
            return super().handle_event( event, global_state, connections )
        return False

    def compute( self ):
        self.values[ "out" ] = self.value

    def draw( self, surface, font ):
        super().draw( surface, font )
        
        if self.editing:
            # --- Draw the input box when editing ---
            input_rect = pygame.Rect( self.rect.centerx - 40, self.rect.centery - 12, 80, 24 )
            pygame.draw.rect( surface, INPUT_BOX_COLOR, input_rect )
            pygame.draw.rect( surface, WHITE, input_rect, 1 )
            
            text_surf = font.render( self.input_text, True, WHITE )
            surface.blit( text_surf, ( input_rect.x + 5, input_rect.y + 5 ) )

            # Blinking cursor
            if pygame.time.get_ticks() % 1000 < 500:
                cursor_pos = input_rect.x + text_surf.get_width() + 8
                pygame.draw.line( surface, WHITE, ( cursor_pos, input_rect.y + 5 ), ( cursor_pos, input_rect.y + 18 ) )
        else:
            # --- Display the value on the node ---
            value_surf = font.render( str( self.value ), True, WHITE )
            value_rect = value_surf.get_rect( center=self.rect.center )
            surface.blit( value_surf, value_rect )

# --- Output nodes ---
class MouseOutputNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y, 120, 100, title="Mouse Output")
        self.add_input("x")
        self.add_input("y")
        self.add_input("btn_L")
        self.add_input("btn_M")
        self.add_input("btn_R")
        self._update_socket_positions()
        
        # Track button states to press/release correctly
        self.button_states = {'left': False, 'middle': False, 'right': False}

    def get_input_value(self, socket_name, default=0):
        """Helper to get a connected input's value."""
        for sock in self.input_sockets:
            if sock['name'] == socket_name and sock.get('connection'):
                source_node = sock['connection']['source_node']
                source_socket_name = sock['connection']['source_socket']['name']
                return source_node.values.get(source_socket_name, default)
        return default

    def compute(self):
        # Get target values from inputs
        target_x = self.get_input_value("x", None)
        target_y = self.get_input_value("y", None)
        
        # Move mouse if x and y are connected
        if target_x is not None and target_y is not None:
            try:
                # Ensure position is within screen bounds
                width, height = pyautogui.size()
                pyautogui.moveTo(
                    max(0, min(int(target_x), width - 1)),
                    max(0, min(int(target_y), height - 1)),
                    duration=0 # Move instantly
                )
            except Exception as e:
                print(f"Could not move mouse: {e}")

        # Handle button presses
        btn_l_active = self.get_input_value("btn_L", 0) > 0.5
        btn_m_active = self.get_input_value("btn_M", 0) > 0.5
        btn_r_active = self.get_input_value("btn_R", 0) > 0.5

        # Left Button
        if btn_l_active and not self.button_states['left']:
            pyautogui.mouseDown(button='left')
            self.button_states['left'] = True
        elif not btn_l_active and self.button_states['left']:
            pyautogui.mouseUp(button='left')
            self.button_states['left'] = False

        # Middle Button
        if btn_m_active and not self.button_states['middle']:
            pyautogui.mouseDown(button='middle')
            self.button_states['middle'] = True
        elif not btn_m_active and self.button_states['middle']:
            pyautogui.mouseUp(button='middle')
            self.button_states['middle'] = False
            
        # Right Button
        if btn_r_active and not self.button_states['right']:
            pyautogui.mouseDown(button='right')
            self.button_states['right'] = True
        elif not btn_r_active and self.button_states['right']:
            pyautogui.mouseUp(button='right')
            self.button_states['right'] = False


class DisplayNode( Node ):
    def __init__( self, x, y ):
        super().__init__( x, y, 100, 60, title="Display" )
        self.add_input( "in" )
        self.display_value = "None"
        self._update_socket_positions()

    def compute( self ):
        # Get value from the input connection
        if self.input_sockets[ 0 ][ 'connection' ]:
            source_node = self.input_sockets[ 0 ][ 'connection' ][ 'source_node' ]
            source_socket_name = self.input_sockets[ 0 ][ 'connection' ][ 'source_socket' ][ 'name' ]
            self.display_value = source_node.values.get( source_socket_name, "None" )
        else:
            self.display_value = "None"

    def draw( self, surface, font ):
        super().draw( surface, font )
        # Display the computed value on the node
        display_text = str( self.display_value )
        if isinstance(self.display_value, float):
             display_text = f"{self.display_value:.3f}" # Format floats nicely

        value_surf = font.render( display_text, True, WHITE )
        value_rect = value_surf.get_rect( center=self.rect.center )
        surface.blit( value_surf, value_rect )
        
class PreviewNode( Node ):
    def __init__( self, x, y ):
        super().__init__( x, y, 100, 60, title="Preview" )
        self.add_input( "in" )
        self.add_output( "out" )
        self.display_value = "None"
        self._update_socket_positions()

    def compute( self ):
        val_a = 0
        # Get value from the input connection
        if self.input_sockets[ 0 ][ 'connection' ]:
            source_node = self.input_sockets[ 0 ][ 'connection' ][ 'source_node' ]
            source_socket_name = self.input_sockets[ 0 ][ 'connection' ][ 'source_socket' ][ 'name' ]
            self.display_value = source_node.values.get( source_socket_name, "None" )
            val_a = source_node.values.get( source_socket_name, 0 )
        else:
            self.display_value = "None"
        
        self.values[ "out" ] = val_a

    def draw( self, surface, font ):
        super().draw( surface, font )
        # Display the computed value on the node
        display_text = str( self.display_value )
        if isinstance(self.display_value, float):
             display_text = f"{self.display_value:.3f}" # Format floats nicely

        value_surf = font.render( display_text, True, WHITE )
        value_rect = value_surf.get_rect( center=self.rect.center )
        surface.blit( value_surf, value_rect )
