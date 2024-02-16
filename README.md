# debt_collection_docs
Policy Lab SP'24

This repository contains code to scrape legal documents from LA County debt collection cases for Policy Lab. The code looks for addresses on the last (6th or 7th) page of the Civil Case Cover Sheet, and looks for the initial demand on the 1st or 2nd page of the Complaint. The addresses are detected ~70% of the time and the initial demands are detected ~67% of the time; however, manual verification is still recommended because there are sometimes errors from the OCR. 

## Quickstart 
### Setup
* Download the legal files you want to scrape over, as well as the csv containing the details on case numbers, file names, etc. (The code assumes the file naming convention and csv matches what we were given in the lab's Box folder.) 
* Make sure you have python, pip, and git installed on your machine.
* [install tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) - this is the Optical Character Recognition (OCR) engine
* [install poppler](https://pdf2image.readthedocs.io/en/latest/installation.html) - this is used for pdf/image conversion library 
* Clone this repo and cd into it

  ```git clone https://github.com/hlu109/debt_collection_docs```

  ```cd debt_collection_docs```
* (Recommended) Create a python virtual environment in the debt_collection_docs folder 
  * if you haven't used virtual environments before, you may need to install the package, e.g. ```pip install virtualenv```
  * Then create the actual virtual environment, e.g. ```virtualenv <venv_name>```
  * Activate the virtual environment, e.g. ```source <venv_name>/bin/activate```
* install the package dependencies

  ```pip install -r requirements.txt```

### Function calls 
The two main functions in this repo are ```extract_all_addresses(input_csv_path, file_dir, output_csv_path)``` and ```extract_all_init_demands(input_csv_path, file_dir, output_csv_path)```. To call either of them, simply pass 
* the path to the original csv containing all the case numbers you want to extract information for
* the path to the folder containing all the actual pdf documents
* the path of a new output csv.

The functions will iterate over all the case numbers, attempt to extract the desired information, and save the results to a new csv. It will also log whether or not the automated extraction passed or failed in the Notes section of the csv, along with an error message if there is one. 

## How it works: 
### Initial demand 
The process to extract the initial demand is fairly straightforward. There appear to be 3 general versions of the Complaint file. One has the demand is present on the first page in the format ```DEMAND: $XXXXX.XX```; another has it on the first page in the format ```PRAYER AMOUNT: $XXXXX.XX```; and another has it on the second page in section 10., which says ```10. Plaintiff prays for judgment for costs of suit; for such relief as is fair, just, and equitable; and for a. damages of: $XXXXX.XX```. (I have not been able to discern a pattern between cases that have different forms.) The extraction process simply checks both the first and second page to see if it can detect each of these patterns using regular expressions (regex); if so, it extracts the string, cleans it up, and returns the float value. 

Note that even though the code is able to detect this pattern ~2/3rds of the time, this doesn't necessarily mean it's all accurate, as there may be OCR errors that we should manually check for. 


### Address 
Automating address extraction is much more complicated. It isn't as easy as running OCR over the entire page containing the address because the page isn't single column; the address fields are located in their own boxes. The address text often gets mixed together with other text from the page, making it very difficult to clean, create a regex pattern, or logic through. 

There are lots of workarounds, but the more successful ones probably all require some amount of computer vision. In the current address extraction algorithm, we use some very basic box detection which someone else has kindly provided open-source (that algorithm is still extremely sensitive and often fails to detect boxes). By specifying the expected sizes for the street, city, state, and zip code boxes, we are able to detect the respective boxes on the form and extract crops of the image containing only that text. Then, when we run OCR over the inividual address field boxes, we are able to extract the address field text pretty cleanly. 


## Alternative strategies and things to improve: 
### For initial demand extraction
* I suspect that a lot of the cases where the algorithm fails to extract the initial demand is because the OCR has some typos and fails to match the regex pattern. We should double check this, and if this is the case, figure out a way to look for strings within a small edit distance.

### For address extraction
* We are still sometimes getting minor OCR errors sneaking in, like diacritics and random punctuation.
  * It should be fairly easy to filter out the random punctuation
  * removing diacritics/restricting the characters to the latin alphabet is a bit of a problem because apparently the tesseract character whitelist only works for the legacy mode, not the newer neural net/LSTM mode, which is what we use in this repo (https://stackoverflow.com/a/49030935/10536083). We could try comparing the performance of the neural net vs legacy-with-whitelist to see what is better. 
* Sometimes the typos are just so bad as to be unrecognizable (e.g. OI instead of CITY) but the actual name of the city is correctly parsed by the OCR. so if we check for the word CITY instead of the city name directly (which we currently do), we're missing out on some files that we could successfully parse. an alternative is to check for the presence of a valid LA city name, which we have a list of. (we would probably also want to check for typos with small edit distance)
* The box detection algorithm is very sensitive to rotation and very finicky to use. we could fork that repo and try to improve it (doable but would take a while)
* Sometimes the box detection algorithm detects the wrong box (e.g. multiple times we have detected the ZIP CODE box when we really wanted the CITY box). An alternative strategy is to find all boxes on a page, then check each one for the text ADDRESS, CITY, STATE, ZIP CODE, etc. However, because the box detection code is so finicky and unreliable, it sometimes sometimes can't find two boxes of the same size on the same page that it should be able to find, thus we might not actually extract all the boxes on the page. Again, this would require improving the actual box detection algorithm. 
