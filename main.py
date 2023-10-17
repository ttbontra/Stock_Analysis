import tkinter as tk
from tkinter import ttk
from forecast import train_and_forecast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StockForecastApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Stock Forecast App")
        self.geometry("800x600")

        self.label = ttk.Label(self, text="Enter Stock Ticker:")
        self.label.pack(pady=20)

        self.ticker_entry = ttk.Entry(self)
        self.ticker_entry.pack(pady=20)

        self.forecast_button = ttk.Button(self, text="Get Forecast", command=self.get_forecast)
        self.forecast_button.pack(pady=20)

    def get_forecast(self):
        ticker = self.ticker_entry.get()
        x_actual, actual, x_predicted, predicted = train_and_forecast(ticker)

        if x_actual is None or actual is None or x_predicted is None or predicted is None:
            ttk.Label(self, text=f"Error fetching or processing data for {ticker}").pack()
            return

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_actual, actual, color='blue', label="Actual Prices")
        ax.plot(x_predicted, predicted, color='red', label="Predicted Prices")
        ax.set_title(f"Stock Forecast for {ticker}")
        ax.legend()

        # Embedding the plot in the GUI
        canvas = FigureCanvasTkAgg(fig, master=self)  
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    app = StockForecastApp()
    app.mainloop()