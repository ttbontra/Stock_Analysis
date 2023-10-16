import tkinter as tk
from tkinter import ttk
import yfinance as yf
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MiniBloombergTerminal(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Mini Bloomberg Terminal")
        self.geometry("1000x600")

        self.create_widgets()

    def create_widgets(self):
        # Top frame for stock symbol entry and predict button
        self.top_frame = ttk.Frame(self)
        self.top_frame.pack(pady=20, fill=tk.X)

        self.label = ttk.Label(self.top_frame, text="Enter Stock Symbol:")
        self.label.grid(row=0, column=0, padx=(10, 5))

        self.stock_symbol_entry = ttk.Entry(self.top_frame)
        self.stock_symbol_entry.grid(row=0, column=1, padx=5)

        self.predict_button = ttk.Button(self.top_frame, text="Fetch & Predict", command=self.start_data_retrieval)
        self.predict_button.grid(row=0, column=2, padx=(5, 10))

        # Split Frame (PanedWindow) to allow resizing
        self.split_frame = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.split_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Left frame for textual stock data
        self.info_frame = ttk.LabelFrame(self.split_frame, text="Stock Info")
        self.split_frame.add(self.info_frame, width=400)

        self.info_text = tk.Text(self.info_frame, wrap=tk.WORD)
        self.info_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Right frame for graph and prediction
        self.right_frame = ttk.Frame(self.split_frame)
        self.split_frame.add(self.right_frame)

        self.canvas_frame = ttk.Frame(self.right_frame)
        self.canvas_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        self.result_label = ttk.Label(self.right_frame, text="")
        self.result_label.pack(pady=20)

    def start_data_retrieval(self):
        threading.Thread(target=self.fetch_and_predict).start()

    def fetch_and_predict(self):
        symbol = self.stock_symbol_entry.get()

        # Fetching stock data
        stock_data = yf.Ticker(symbol)

        # Display stock info
        stock_info = stock_data.info
        display_info = f"Name: {stock_info.get('longName', 'N/A')}\n"
        display_info += f"Sector: {stock_info.get('sector', 'N/A')}\n"
        display_info += f"Industry: {stock_info.get('industry', 'N/A')}\n"
        display_info += f"Description: {stock_info.get('longBusinessSummary', 'N/A')}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, display_info)

        # Fetching current stock price
        current_price = stock_data.history(period="1d")["Close"][0]
        # Dummy prediction logic
        predicted_price = current_price * 1.01

        self.result_label.config(text=f"Current Price: ${current_price:.2f}\nPredicted Price: ${predicted_price:.2f}")

        # Plotting historical data
        hist_data = stock_data.history(period="1y")
        self.plot_data(hist_data["Close"].tolist())

    def plot_data(self, data):
        # Clear previous plots
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(data)
        ax.set_title("1 Year Historical Stock Price")
        ax.grid(True)

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

if __name__ == "__main__":
    app = MiniBloombergTerminal()
    app.mainloop()
