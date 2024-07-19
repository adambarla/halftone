# Runnning the example

First install python and requirements as described in [Installation section in the main README file](../README.md#Installation) .

Then run the following commands:
```bash
conda activate halftone
pip install notebook
jupyter notebook
```

This will open a new tab in your browser with jupyter. You should see the folder structure of the project. Navigate to the `example` folder and open the `jupyter_example.ipynb` notebook.

Running the notebook will generate a `svg` file in the `example` folder. You can open it in your browser to see the result. It should look like this:

![Example halftone](tatry_cmyk.svg)

## Customization

You can customize the output by changing the parameters of the `generate_halftone` function.

Besides paths to the input and output files, you can change the size of the paper, maximum dot size, line thickness, opacity of the colors and more.

While having cursor on the function you can use `Shift + Tab` to see the documentation. 
You can also see the documentation in the [main README file](../README.md#Usage) or directly in the code in the `src/halftone.py` file.