import tkinter as tk
from tkinter import ttk
import yfinance as yf
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StockPricePredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Stock Price Predictor")

        self.label = ttk.Label(self, text="Enter Stock Symbol:")
        self.label.pack(pady=20)

        self.stock_symbol_entry = ttk.Entry(self)
        self.stock_symbol_entry.pack(pady=20)

        self.predict_button = ttk.Button(self, text="Predict", command=self.start_prediction)
        self.predict_button.pack(pady=20)

        self.result_label = ttk.Label(self, text="")
        self.result_label.pack(pady=20)

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(pady=20, fill=tk.BOTH, expand=True)

    def start_prediction(self):
        threading.Thread(target=self.predict_stock_price).start()

    def predict_stock_price(self):
        symbol = self.stock_symbol_entry.get()

        # Fetching current stock price using yfinance
        stock_data = yf.Ticker(symbol)
        current_price = stock_data.history(period="1d")["Close"][0]

        # Dummy prediction logic
        predicted_price = current_price * 1.01

        self.result_label.config(text=f"Current Price: ${current_price:.2f}\nPredicted Price: ${predicted_price:.2f}")

        # Plotting the data
        self.plot_data([current_price, predicted_price])

    def plot_data(self, data):
        # Clear the frame
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(["Current", "Predicted"], data)
        ax.set_title("Stock Price Prediction")
        ax.grid(True)

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

if __name__ == "__main__":
    app = StockPricePredictorApp()
    app.mainloop()
