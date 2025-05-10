import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import numpy as np
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the matcher modules - will implement these separately
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from methods import clip_matcher, dino_matcher, vgg_matcher, orb_matcher, ssim_matcher, imagebind_matcher


class ImageMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Matching Methods Comparison")
        self.root.geometry("1200x800")

        # Folder paths
        self.folder1 = None
        self.folder2 = None

        # Available matching methods
        self.methods = {
            "CLIP": clip_matcher.match_images,
            "DINO/DINOv2": dino_matcher.match_images,
            "VGG Features": vgg_matcher.match_images,
            "ORB": orb_matcher.match_images,
            "SSIM": ssim_matcher.match_images,
            "ImageBind": imagebind_matcher.match_images
        }

        # Results storage
        self.results = {}
        self.current_method = None
        self.current_match_index = 0

        # UI components
        self.setup_ui()

    def setup_ui(self):
        # Create a frame for folder selection
        folder_frame = ttk.Frame(self.root)
        folder_frame.pack(pady=10, fill=tk.X, padx=10)

        # Folder selection buttons and path display
        ttk.Button(folder_frame, text="Select Camera 1 Folder", command=self.select_folder1).grid(row=0, column=0,
                                                                                                  padx=5, pady=5)
        self.folder1_label = ttk.Label(folder_frame, text="No folder selected")
        self.folder1_label.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Button(folder_frame, text="Select Camera 2 Folder", command=self.select_folder2).grid(row=1, column=0,
                                                                                                  padx=5, pady=5)
        self.folder2_label = ttk.Label(folder_frame, text="No folder selected")
        self.folder2_label.grid(row=1, column=1, sticky='w', padx=5)

        # Method selection with checkboxes
        method_frame = ttk.LabelFrame(self.root, text="Select Matching Methods")
        method_frame.pack(pady=10, fill=tk.X, padx=10)

        self.method_vars = {}
        for i, method in enumerate(self.methods.keys()):
            var = tk.BooleanVar(value=True)
            self.method_vars[method] = var
            ttk.Checkbutton(method_frame, text=method, variable=var).grid(row=i // 3, column=i % 3, sticky='w', padx=15,
                                                                          pady=5)

        # Top-N selection
        config_frame = ttk.Frame(self.root)
        config_frame.pack(pady=5, fill=tk.X, padx=10)

        ttk.Label(config_frame, text="Number of top matches:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.top_n_var = tk.StringVar(value="5")
        ttk.Spinbox(config_frame, from_=1, to=20, textvariable=self.top_n_var, width=5).grid(row=0, column=1,
                                                                                             sticky='w', padx=5)

        # Run button
        ttk.Button(config_frame, text="Run Comparison", command=self.run_comparison).grid(row=0, column=2, padx=20)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # Method tabs for results
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Tab for performance comparison
        self.perf_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.perf_tab, text="Performance Comparison")

        # Performance metrics display will be added here when results are available

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def select_folder1(self):
        folder = filedialog.askdirectory(title="Select Camera 1 Folder")
        if folder:
            self.folder1 = folder
            self.folder1_label.config(text=f"{os.path.basename(folder)} ({len(self.get_image_files(folder))} images)")

    def select_folder2(self):
        folder = filedialog.askdirectory(title="Select Camera 2 Folder")
        if folder:
            self.folder2 = folder
            self.folder2_label.config(text=f"{os.path.basename(folder)} ({len(self.get_image_files(folder))} images)")

    def get_image_files(self, folder):
        """Get all image files in a folder."""
        extensions = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp')
        return [os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and
                f.lower().endswith(extensions)]

    def run_comparison(self):
        """Run the selected matching methods."""
        if not self.folder1 or not self.folder2:
            self.status_var.set("Please select both folders first")
            return

        # Clear previous results
        for tab in self.notebook.tabs():
            if tab != self.perf_tab:
                self.notebook.forget(tab)

        self.results = {}

        # Get selected methods
        selected_methods = [method for method, var in self.method_vars.items() if var.get()]
        if not selected_methods:
            self.status_var.set("Please select at least one matching method")
            return

        # Get top-n value
        try:
            top_n = int(self.top_n_var.get())
        except ValueError:
            self.status_var.set("Please enter a valid number for top matches")
            return

        # Disable UI during processing
        self.disable_ui()

        # Start processing thread
        thread = threading.Thread(target=self._process_methods,
                                  args=(selected_methods, top_n))
        thread.daemon = True
        thread.start()

    def _process_methods(self, selected_methods, top_n):
        """Process all selected methods in a background thread."""
        try:
            total_methods = len(selected_methods)
            performance_data = {'Method': [], 'Time (s)': [], 'Memory (MB)': []}

            for i, method_name in enumerate(selected_methods):
                self.status_var.set(f"Processing with {method_name}...")
                self.progress_var.set((i / total_methods) * 100)

                # Get the matcher function
                matcher_func = self.methods[method_name]

                # Process the images
                start_time = time.time()
                result = matcher_func(self.folder1, self.folder2, top_n=top_n)
                elapsed_time = time.time() - start_time

                # Store results
                self.results[method_name] = {
                    'matches': result['matches'],
                    'processing_time': elapsed_time,
                    'memory_usage': result.get('memory_usage', 0)
                }

                # Collect performance data
                performance_data['Method'].append(method_name)
                performance_data['Time (s)'].append(elapsed_time)
                performance_data['Memory (MB)'].append(result.get('memory_usage', 0))

                # Create a tab for this method
                self.root.after(0, lambda m=method_name: self.create_method_tab(m))

            # Update progress
            self.progress_var.set(100)

            # Create performance comparison visualizations
            self.root.after(0, lambda: self.create_performance_comparison(performance_data))

            # Enable UI
            self.root.after(0, self.enable_ui)
            self.status_var.set("Comparison completed")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.enable_ui()

    def create_method_tab(self, method_name):
        """Create a tab to display results for a specific method."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=method_name)

        # Create navigation frame
        nav_frame = ttk.Frame(tab)
        nav_frame.pack(fill=tk.X, pady=5)

        # Navigation controls
        matches = self.results[method_name]['matches']

        if not matches:
            ttk.Label(tab, text="No matches found").pack(pady=20)
            return

        self.match_index_vars = {}
        self.match_index_vars[method_name] = tk.IntVar(value=0)

        ttk.Button(nav_frame, text="Previous",
                   command=lambda: self.navigate_matches(method_name, -1)).pack(side=tk.LEFT, padx=5)

        ttk.Label(nav_frame, text="Match:").pack(side=tk.LEFT)
        match_spinner = ttk.Spinbox(nav_frame, from_=1, to=len(matches),
                                    textvariable=self.match_index_vars[method_name],
                                    width=5, command=lambda: self.on_match_select(method_name))
        match_spinner.pack(side=tk.LEFT, padx=5)
        ttk.Label(nav_frame, text=f"of {len(matches)}").pack(side=tk.LEFT)

        ttk.Button(nav_frame, text="Next",
                   command=lambda: self.navigate_matches(method_name, 1)).pack(side=tk.LEFT, padx=5)

        # Create the match display frame
        display_frame = ttk.Frame(tab)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Store the frame reference for updating
        self.results[method_name]['display_frame'] = display_frame

        # Display the first match
        self.display_match(method_name, 0)

    def navigate_matches(self, method_name, direction):
        """Navigate to the previous or next match."""
        matches = self.results[method_name]['matches']
        if not matches:
            return

        # Get the current index
        current = self.match_index_vars[method_name].get() - 1

        # Calculate the new index
        new_index = (current + direction) % len(matches)

        # Update the index variable
        self.match_index_vars[method_name].set(new_index + 1)

        # Display the match
        self.display_match(method_name, new_index)

    def on_match_select(self, method_name):
        """Handle match selection via spinbox."""
        try:
            # Get the selected index (1-based)
            selected = self.match_index_vars[method_name].get()

            # Adjust to 0-based for internal use
            index = selected - 1

            # Validate range
            matches = self.results[method_name]['matches']
            if 0 <= index < len(matches):
                # Display the match
                self.display_match(method_name, index)
        except Exception as e:
            print(f"Error in match selection: {e}")

    def display_match(self, method_name, index):
        """Display a specific match."""
        # Get the match information
        matches = self.results[method_name]['matches']
        if not matches or index >= len(matches):
            return

        match = matches[index]
        img1_path, img2_path, similarity = match

        # Clear the display frame
        display_frame = self.results[method_name]['display_frame']
        for widget in display_frame.winfo_children():
            widget.destroy()

        # Create frames for the two images
        left_frame = ttk.Frame(display_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        right_frame = ttk.Frame(display_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Add image labels
        ttk.Label(left_frame, text=f"Camera 1: {os.path.basename(img1_path)}").pack(pady=5)
        ttk.Label(right_frame, text=f"Camera 2: {os.path.basename(img2_path)}").pack(pady=5)

        # Load and display the images
        try:
            # Left image
            img1 = Image.open(img1_path)
            img1.thumbnail((500, 500))  # Resize for display
            photo1 = ImageTk.PhotoImage(img1)
            img1_label = ttk.Label(left_frame, image=photo1)
            img1_label.image = photo1  # Keep a reference
            img1_label.pack(pady=5)

            # Right image
            img2 = Image.open(img2_path)
            img2.thumbnail((500, 500))  # Resize for display
            photo2 = ImageTk.PhotoImage(img2)
            img2_label = ttk.Label(right_frame, image=photo2)
            img2_label.image = photo2  # Keep a reference
            img2_label.pack(pady=5)

            # Display similarity score
            ttk.Label(display_frame,
                      text=f"Similarity Score: {similarity:.4f}",
                      font=('Arial', 12, 'bold')).pack(side=tk.BOTTOM, pady=10)

        except Exception as e:
            ttk.Label(display_frame, text=f"Error loading images: {str(e)}").pack(pady=20)

    def create_performance_comparison(self, performance_data):
        """Create visualizations for performance comparison."""
        # Clear previous content
        for widget in self.perf_tab.winfo_children():
            widget.destroy()

        if not performance_data['Method']:
            ttk.Label(self.perf_tab, text="No performance data available").pack(pady=20)
            return

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot  processing time
        methods = performance_data['Method']
        times = performance_data['Time (s)']
        ax1.bar(methods, times)
        ax1.set_title('Processing Time (seconds)')
        ax1.set_ylabel('Time (s)')
        ax1.tick_params(axis='x', rotation=45)

        # Plot memory usage
        memory = performance_data['Memory (MB)']
        ax2.bar(methods, memory)
        ax2.set_title('Memory Usage (MB)')
        ax2.set_ylabel('Memory (MB)')
        ax2.tick_params(axis='x', rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Add the plot to the tab
        canvas = FigureCanvasTkAgg(fig, master=self.perf_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a summary table
        table_frame = ttk.Frame(self.perf_tab)
        table_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(table_frame, text="Method", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(table_frame, text="Time (s)", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Memory (MB)", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2)

        for i, method in enumerate(methods):
            ttk.Label(table_frame, text=method).grid(row=i + 1, column=0, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{times[i]:.2f}").grid(row=i + 1, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{memory[i]:.2f}").grid(row=i + 1, column=2, padx=5, pady=2)

    def on_tab_change(self, event):
        """Handle tab change event."""
        selected_tab = self.notebook.select()
        tab_name = self.notebook.tab(selected_tab, "text")
        self.current_method = tab_name if tab_name != "Performance Comparison" else None

    def disable_ui(self):
        """Disable UI elements during processing."""
        for widget in self.root.winfo_children():
            if isinstance(widget, (ttk.Button, ttk.Checkbutton, ttk.Spinbox)):
                widget.configure(state='disabled')

    def enable_ui(self):
        """Enable UI elements after processing."""
        for widget in self.root.winfo_children():
            if isinstance(widget, (ttk.Button, ttk.Checkbutton, ttk.Spinbox)):
                widget.configure(state='normal')


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMatcherApp(root)
    root.mainloop()