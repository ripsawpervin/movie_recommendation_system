"""Run with: python main.py"""
import tkinter as tk
from tkinter import messagebox
def main():
    root = tk.Tk()
    root.withdraw()
    try:
        from data_manager import DataManager
        from recommender import MovieRecommender
        from gui import MovieApp
        data = DataManager()
        engine = MovieRecommender(data)
        MovieApp(root, data, engine)
    except Exception as error:
        messagebox.showerror('Startup problem', f'{error}\n\nCheck the dataset and install requirements.txt.', parent=root)
        root.destroy()
        return
    root.deiconify()
    if data.warnings:
        messagebox.showwarning('Saved lists', '\n'.join(data.warnings), parent=root)
    root.mainloop()
if __name__ == '__main__':
    main()
