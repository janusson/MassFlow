def process_spectra_generator(input_file):
    # 1. Load ONE item
    for spectrum in load_file(input_file):
        cleaned = clean(spectrum)
        # 2. Hand it over immediately, then forget it
        yield cleaned # hand it over, pause the pump

# Main Program
# Open the CSV file FIRST
with open("results.csv", "w") as f:
    # 3. The Loop "pulls" one item at a time
    for single_result in process_spectra_generator("huge_file.mgf"):
        # 4. Write it to disk immediately
        f.write(f"{single_result.name},{single_result.score}\n")
        # 5. The loop repeats. 'single_result' is overwritten. RAM is cleared.