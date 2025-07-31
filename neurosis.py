import pygame
import sys
import nodes
import inspect


# --- Colors ---
WHITE = ( 255, 255, 255 )
BLACK = ( 0, 0, 0 )
GREY = ( 150, 150, 150 )
NODE_BODY_COLOR = ( 100, 100, 120 )
NODE_BORDER_COLOR = ( 200, 200, 220 )
CONNECTION_COLOR = ( 200, 200, 100 )
SOCKET_COLOR = ( 50, 150, 250 )
INPUT_BOX_COLOR = ( 30, 30, 40 )

def get_node_classes():
    node_classes = {}
    for name, obj in inspect.getmembers(nodes):
        if inspect.isclass(obj) and name.endswith("Node"):
            # Skip the base Node class
            if name == "Node":
                continue
            # Use a readable label, e.g. "Add" instead of "AddNode"
            label = name.replace("Node", "")
            node_classes[label] = obj
    return node_classes

class ContextMenu:
    # --- Right-click context menu with scrolling ---
    def __init__(self, pos, options, all_nodes):
        self.pos = pos
        self.options = options
        self.all_nodes = all_nodes
        self.width = 180
        self.item_height = 25
        self.visible_items = 10  # Number of items visible at once
        self.height = min(len(options), self.visible_items) * self.item_height
        self.menu_rect = pygame.Rect(pos[0], pos[1], self.width, self.height)
        self.action_to_perform = None
        self.click_pos = pos
        self.scroll_offset = 0
        self.max_scroll = max(0, len(options) - self.visible_items)
        self.option_keys = list(options.keys())

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                self.scroll_offset = max(0, self.scroll_offset - 1)
            elif event.button == 5:  # Scroll down
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + 1)
            elif event.button == 1:  # Left-click
                mx, my = event.pos
                if self.menu_rect.collidepoint(mx, my):
                    idx = (my - self.menu_rect.y) // self.item_height + self.scroll_offset
                    if 0 <= idx < len(self.option_keys):
                        action = self.options[self.option_keys[idx]]
                        if callable(action):
                            new_node = action(self.pos)
                            self.all_nodes.append(new_node)
                        return True
                else:
                    return True  # Click outside closes menu
        elif event.type == pygame.MOUSEBUTTONUP:
            pass
        return False

    def draw(self, surface, font):
        pygame.draw.rect(surface, (50, 50, 50), self.menu_rect)
        pygame.draw.rect(surface, (150, 150, 150), self.menu_rect, 1)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for i in range(self.visible_items):
            idx = i + self.scroll_offset
            if idx >= len(self.option_keys):
                break
            item_rect = pygame.Rect(
                self.menu_rect.x, self.menu_rect.y + i * self.item_height, self.width, self.item_height
            )
            # Highlight on hover
            if item_rect.collidepoint(mouse_x, mouse_y):
                pygame.draw.rect(surface, (80, 80, 100), item_rect)
            text_surf = font.render(self.option_keys[idx], True, WHITE)
            surface.blit(text_surf, (item_rect.x + 5, item_rect.y + 5))
        # Draw scroll indicators if needed
        if self.max_scroll > 0:
            if self.scroll_offset > 0:
                pygame.draw.polygon(surface, WHITE, [
                    (self.menu_rect.right - 15, self.menu_rect.y + 5),
                    (self.menu_rect.right - 5, self.menu_rect.y + 5),
                    (self.menu_rect.right - 10, self.menu_rect.y + 12)
                ])
            if self.scroll_offset < self.max_scroll:
                pygame.draw.polygon(surface, WHITE, [
                    (self.menu_rect.right - 15, self.menu_rect.bottom - 5),
                    (self.menu_rect.right - 5, self.menu_rect.bottom - 5),
                    (self.menu_rect.right - 10, self.menu_rect.bottom - 12)
                ])

# --- Main Application ---
def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont( None, 24 )
    small_font = pygame.font.SysFont( None, 20 )

    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    screen = pygame.display.set_mode( ( SCREEN_WIDTH, SCREEN_HEIGHT ) )
    pygame.display.set_caption( "Neurosis" )

    nodes = [ # --- Default nodes on opening ---

    ]
    connections = []

    global_connection_state = {
        'is_drawing_connection': False,
        'connection_start_node': None,
        'connection_start_socket': None,
    }
    
    context_menu = None
    
    # --- Track which node is being edited ---
    editing_node = None

    running = True
    clock = pygame.time.Clock()

    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # --- Determine which node is being edited ---
        editing_node = None
        for n in nodes:
            if getattr(n, 'editing', False):
                editing_node = n
                break

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # --- Pass keyboard events to the editing node FIRST ---
            if editing_node:
                editing_node.handle_event( event, global_connection_state, connections )
                # If a click happens, check if it's outside the editing node to close it
                if event.type == pygame.MOUSEBUTTONDOWN and not editing_node.rect.collidepoint( event.pos ):
                    editing_node.editing = False
                    editing_node.input_text = str( editing_node.value ) # revert
                continue # Skip other handlers if we are editing

            # --- DELETE NODE with Delete Key ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    node_to_delete = None
                    for node in nodes:
                        if node.rect.collidepoint( mouse_pos ):
                            node_to_delete = node
                            break # Found the node to delete
                    
                    if node_to_delete:
                        # Remove connections associated with this node
                        connections[:] = [ c for c in connections if c[ 'source_node' ] != node_to_delete and c[ 'target_node' ] != node_to_delete ]
                        
                        # Unlink from any nodes that were targeting it
                        for n in nodes:
                            if n == node_to_delete: continue
                            for s in n.input_sockets:
                                if s[ 'connection' ] and s[ 'connection' ][ 'source_node' ] == node_to_delete:
                                    s[ 'connection' ] = None
                        
                        nodes.remove( node_to_delete )
                        continue # Event handled

            # --- Context Menu Handling ---
            if context_menu:
                if context_menu.handle_event( event ):
                    context_menu = None # Close menu after action
                    continue # Skip other event handling

            # --- Finalize Connection ---
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and global_connection_state[ 'is_drawing_connection' ]:
                target_found = False
                for node in nodes:
                    for sock in node.input_sockets:
                        if sock[ 'rect' ].collidepoint( event.pos ) and sock[ 'connection' ] is None:
                            # Create connection
                            new_conn = {
                                'source_node': global_connection_state[ 'connection_start_node' ],
                                'source_socket': global_connection_state[ 'connection_start_socket' ],
                                'target_node': node,
                                'target_socket': sock
                            }
                            connections.append( new_conn )
                            sock[ 'connection' ] = new_conn # Link locally
                            target_found = True
                            break
                    if target_found: break
                
                # Reset connection drawing state
                global_connection_state[ 'is_drawing_connection' ] = False
                global_connection_state[ 'connection_start_node' ] = None
                global_connection_state[ 'connection_start_socket' ] = None
                continue

            # --- Open Context Menu ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                # Prevent menu if clicking on a node's socket
                on_socket = False
                for node in nodes:
                    for sock in node.input_sockets + node.output_sockets:
                        if sock['rect'].collidepoint(event.pos):
                            on_socket = True
                            break
                    if on_socket: break

                if not on_socket:
                    node_classes = get_node_classes()
                    context_menu = ContextMenu(
                        event.pos,
                        {label: (lambda pos, cls=cls: cls(pos[0], pos[1])) for label, cls in node_classes.items()},
                        nodes
                    )
                    continue

            # --- Pass events to nodes ---
            for node in reversed( nodes ):
                if node.handle_event( event, global_connection_state, connections ):
                    break

        # --- Update & Compute ---
        # A simple, iterative computation model. For complex graphs, a topological sort would be needed.
        for _ in range( len( nodes ) ): # Iterate a few times to propagate changes
            for node in nodes:
                node.compute()

        # --- Drawing ---
        screen.fill( GREY )

        # Draw established connections
        for conn in connections:
            start_pos = conn[ 'source_socket' ][ 'pos' ]
            end_pos = conn[ 'target_socket' ][ 'pos' ]
            pygame.draw.line( screen, CONNECTION_COLOR, start_pos, end_pos, 2 )
            pygame.draw.aaline( screen, WHITE, start_pos, end_pos )

        # Draw temporary connection line
        if global_connection_state[ 'is_drawing_connection' ]:
            start_pos = global_connection_state[ 'connection_start_socket' ][ 'pos' ]
            pygame.draw.line( screen, CONNECTION_COLOR, start_pos, mouse_pos, 3 )

        # Draw all nodes
        for node in nodes:
            node.draw( screen, font )
        
        # Draw context menu if active
        if context_menu:
            context_menu.draw( screen, small_font )

        # --- Update Display ---
        pygame.display.flip()
        clock.tick( 60 )

    # --- Cleanup ---
    pygame.font.quit()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()