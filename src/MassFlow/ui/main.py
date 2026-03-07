"""
Graphical User Interface (GUI) entry point for MassFlow.

This module implements the main application window using CustomTkinter. It provides
an interactive interface for users to configure and execute the MassFlow annotation
pipeline without using the command line. It features file selection dialogs,
configuration of basic parameters, and a real-time log console to monitor execution
progress.
"""

import logging
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from MassFlow import __version__
from MassFlow.config import InputConfig, MassFlowConfig, ProjectConfig
from MassFlow.workflow import run_annotation_pipeline

# Prevent Numba threading issues on macOS
os.environ["NUMBA_NUM_THREADS"] = "1"

# Setup Logger
logger = logging.getLogger("MassFlow")
logger.setLevel(logging.INFO)


class TextHandler(logging.Handler):
    """
    A logging handler that directs log output to a Tkinter text widget.

    This handler formats log records and appends them to a specified text
    widget, changing the text color based on the log level (e.g., red for
    errors). It ensures thread safety by scheduling UI updates on the main
    GUI thread.
    """

    def __init__(self, text_widget: ctk.CTkTextbox | tk.Text):
        """
        Initialize the TextHandler.

        Parameters
        ----------
        text_widget : customtkinter.CTkTextbox or tkinter.Text
            The text widget where log messages will be displayed.
        """
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        self.text_widget.configure(state="disabled")
        self.text_widget.tag_config("INFO", foreground="white")
        self.text_widget.tag_config("WARNING", foreground="orange")
        self.text_widget.tag_config("ERROR", foreground="red")
        self.text_widget.tag_config("CRITICAL", foreground="red", underline=1)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a formatted log record to the text widget.

        This method formats the log message and schedules its insertion into
        the text widget using Tkinter's `after` method to ensure thread safety
        when logging from background threads.

        Parameters
        ----------
        record : logging.LogRecord
            The log record containing the message and log level.

        Returns
        -------
        None
        """
        msg = self.format(record)

        def append():
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, msg + "\n", record.levelname)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")

        # Schedule the update on the main GUI thread
        self.text_widget.after(0, append)


class AnnotationFrame(ctk.CTkFrame):
    """
    A CustomTkinter frame providing the user interface for the annotation workflow.

    This frame contains input fields and buttons for selecting experimental
    data files, reference libraries, and output directories. It also includes
    the logic to validate inputs and trigger the annotation pipeline in a
    separate background thread.
    """

    def __init__(self, master: Any, **kwargs: Any):
        """
        Initialize the AnnotationFrame layout and widgets.

        Parameters
        ----------
        master : Any
            The parent widget or main application window.
        **kwargs : Any
            Additional keyword arguments passed to the CTkFrame constructor.
        """
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(1, weight=1)

        # Title
        self.label_title = ctk.CTkLabel(
            self,
            text="Spectral Annotation Pipeline",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.label_title.grid(
            row=0, column=0, columnspan=3, padx=20, pady=(20, 10), sticky="ew"
        )

        # Experimental File Selection
        self.label_experimental = ctk.CTkLabel(self, text="Experimental File:")
        self.label_experimental.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.entry_experimental = ctk.CTkEntry(
            self, placeholder_text="Path to .mzML, .mgf, or .msp"
        )
        self.entry_experimental.grid(row=1, column=1, padx=20, pady=10, sticky="ew")

        self.btn_experimental = ctk.CTkButton(
            self, text="Browse", command=self.browse_experimental, width=100
        )
        self.btn_experimental.grid(row=1, column=2, padx=20, pady=10)

        # Reference Library Selection
        self.label_reference = ctk.CTkLabel(self, text="Reference Library:")
        self.label_reference.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        self.entry_reference = ctk.CTkEntry(
            self, placeholder_text="Path to .msp or .mgf"
        )
        self.entry_reference.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

        self.btn_reference = ctk.CTkButton(
            self, text="Browse", command=self.browse_reference, width=100
        )
        self.btn_reference.grid(row=2, column=2, padx=20, pady=10)

        # Output Directory Selection
        self.label_output = ctk.CTkLabel(self, text="Output Directory:")
        self.label_output.grid(row=3, column=0, padx=20, pady=10, sticky="w")

        self.entry_output = ctk.CTkEntry(self, placeholder_text="Select output folder")
        self.entry_output.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

        self.btn_output = ctk.CTkButton(
            self, text="Browse", command=self.browse_output, width=100
        )
        self.btn_output.grid(row=3, column=2, padx=20, pady=10)

        # Run Button
        self.btn_run = ctk.CTkButton(
            self,
            text="Run Annotation",
            command=self.start_annotation_thread,
            fg_color="green",
            hover_color="darkgreen",
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.btn_run.grid(row=4, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

    def browse_experimental(self) -> None:
        """
        Open a file dialog to select the experimental mass spectrometry data file.

        Updates the corresponding entry widget with the selected file path.

        Returns
        -------
        None
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Spectral Files", "*.mzML *.mgf *.msp"), ("All Files", "*.*")]
        )
        if file_path:
            self.entry_experimental.delete(0, tk.END)
            self.entry_experimental.insert(0, file_path)

    def browse_reference(self) -> None:
        """
        Open a file dialog to select the reference library file.

        Updates the corresponding entry widget with the selected file path.

        Returns
        -------
        None
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Library Files", "*.msp *.mgf"), ("All Files", "*.*")]
        )
        if file_path:
            self.entry_reference.delete(0, tk.END)
            self.entry_reference.insert(0, file_path)

    def browse_output(self) -> None:
        """
        Open a directory dialog to select the output destination.

        Updates the corresponding entry widget with the selected directory path.

        Returns
        -------
        None
        """
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, dir_path)

    def start_annotation_thread(self) -> None:
        """
        Validate inputs and start the annotation pipeline in a background thread.

        This method retrieves the paths from the input fields, ensures all required
        values are present, disables the run button to prevent multiple executions,
        and spawns a daemon thread to run the `run_pipeline` method. This prevents
        the heavy processing from blocking the GUI event loop.

        Returns
        -------
        None
        """
        experimental = self.entry_experimental.get()
        reference = self.entry_reference.get()
        output = self.entry_output.get()

        if not all([experimental, reference, output]):
            messagebox.showerror(
                "Error", "Please select all required files and folders."
            )
            return

        self.btn_run.configure(state="disabled", text="Running...")

        thread = threading.Thread(
            target=self.run_pipeline,
            args=(experimental, reference, output),
            daemon=True,
        )
        thread.start()

    def run_pipeline(self, experimental: str, reference: str, output: str) -> None:
        """
        Execute the MassFlow annotation pipeline.

        This method constructs the necessary configuration objects (`ProjectConfig`,
        `InputConfig`, and `MassFlowConfig`) and calls the core `run_annotation_pipeline`
        function. It manages success and error states by displaying message boxes and
        resetting the run button upon completion.

        Parameters
        ----------
        experimental : str
            The file path to the experimental mass spectrometry data.
        reference : str
            The file path to the reference spectral library.
        output : str
            The directory path where the annotation results will be saved.

        Returns
        -------
        None
        """
        try:
            logger.info("Starting annotation pipeline...")
            logger.info(f"Experimental: {experimental}")
            logger.info(f"Reference: {reference}")
            logger.info(f"Output: {output}")

            # Construct Configuration
            config = MassFlowConfig(
                project=ProjectConfig(output_directory=Path(output)),
                input=InputConfig(
                    file_path=Path(experimental),
                    reference_library=Path(reference),
                ),
            )

            run_annotation_pipeline(config)

            logger.info("Pipeline completed successfully!")
            self.after(
                0, lambda: messagebox.showinfo("Success", "Annotation complete!")
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.after(
                0, lambda: messagebox.showerror("Error", f"Pipeline failed:\n{e}")
            )

        finally:
            self.after(
                0, lambda: self.btn_run.configure(state="normal", text="Run Annotation")
            )


class App(ctk.CTk):
    """
    Main application window for the MassFlow GUI.

    This class sets up the main window, instantiates the annotation configuration
    frame, and provides a scrollable text console for viewing real-time log output.
    """

    def __init__(self):
        """
        Initialize the main window, layout, and logging console.
        """
        super().__init__()

        self.title(f"MassFlow v{__version__}")
        self.geometry("900x600")

        # Configure Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Annotation Frame
        self.grid_rowconfigure(1, weight=1)  # Log Console

        # Annotation Frame
        self.annotation_frame = AnnotationFrame(self)
        self.annotation_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Log Console
        self.log_label = ctk.CTkLabel(self, text="Console Output", anchor="w")
        self.log_label.grid(row=1, column=0, padx=20, pady=(0, 5), sticky="w")

        self.log_textbox = ctk.CTkTextbox(self, activate_scrollbars=True)
        self.log_textbox.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Setup Logging to Textbox
        text_handler = TextHandler(self.log_textbox)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)

        # Also log to stdout for debugging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info("Welcome to MassFlow GUI!")


def main() -> None:
    """
    Initialize and run the CustomTkinter MassFlow GUI application.

    Sets the appearance mode and color theme, initializes the main App
    instance, and starts the Tkinter event loop.

    Returns
    -------
    None
    """
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
