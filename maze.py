import numpy as np
from PIL import Image
from collections import deque

img_path = 'MAZE_0.png'
img_file = Image.open(img_path)
gray_img = img_file.convert('L')
maze_array = np.array(gray_img) > 128

def find_openings(maze):
    """Find entrance/exit on borders"""
    h, w = maze.shape
    
    # Check top and bottom rows
    top_openings = [(0, j) for j in range(w) if maze[0, j]]
    bottom_openings = [(h-1, j) for j in range(w) if maze[h-1, j]]
    
    # Check left and right columns
    left_openings = [(i, 0) for i in range(h) if maze[i, 0]]
    right_openings = [(i, w-1) for i in range(h) if maze[i, w-1]]
    
    all_openings = top_openings + bottom_openings + left_openings + right_openings
    return all_openings

openings = find_openings(maze_array)
print(f"Found {len(openings)} border openings: {openings[:5]}...")  # Show first 5

# BFS solver
def solve_maze(maze, start, end):
    """BFS to find shortest path"""
    visited = np.zeros_like(maze, dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start] = True
    parent[start] = None
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        y, x = current
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            # Check bounds and if passable and not visited
            if (0 <= ny < maze.shape[0] and 
                0 <= nx < maze.shape[1] and 
                maze[ny, nx] and 
                not visited[ny, nx]):
                
                visited[ny, nx] = True
                parent[(ny, nx)] = current
                queue.append((ny, nx))
    
    return None  # No path found

# Assuming first and last openings are start/end
if len(openings) >= 2:
    start = openings[0]
    end = openings[-1]
    
    print(f"Solving from {start} to {end}...")
    path = solve_maze(maze_array, start, end)
    
    if path:
        print(f"Found path with {len(path)} steps!")
        
        # Visualize the solution
        solution_img = img_file.convert('RGB')
        pixels = solution_img.load()
        
        for y, x in path:
            pixels[x, y] = (255, 0, 0)  # Red path
        
        solution_img.save('maze_solution.png')
        print("Saved to maze_solution.png")
    else:
        print("No path found!")