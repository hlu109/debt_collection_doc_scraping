# debt_collection_doc_scraping
Policy Lab SP'24

This repository contains code to scrape legal documents from LA County debt collection cases for Policy Lab. The code looks for addresses on the last (usually 6th, very rarely 7th) page of the Civil Case Cover Sheet, and looks for the initial demand on the first 3 pages of the Complaint. 

The addresses are detected ~70% of the time and the initial demands are detected ~90% of the time. However, manual verification is still recommended for address extraction because there are sometimes errors from the Optical Character Recognition (OCR). (Out of a sample of 450 cases, 330 addresses were extracted, with 20 of the addresses having OCR issues of various sorts.) Also, the cases in which this script fails to extract initial demand often share similar characteristics, which make their exclusion non-random and systematic. 

## Quickstart 
### Setup
* Download the legal files you want to scrape over, as well as the csv containing the details on case numbers, file names, etc. (The code assumes the file naming convention and csv matches what we were given in the lab's Box folder.) 
* Make sure you have python, pip, and git installed on your machine.
* [install tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) - this is the OCR engine
  * Make sure to add the tesseract executable to your PATH. You can test if it's there by calling `tesseract` in your command line to see if it returns the help text or crashes.
  * If you don't have the tesseract executable in your PATH, then you need to find the path to the executable; when you clone this repo, go into the ```doc_scraping.py``` file, uncomment the code on line 17 that says ```pytesseract.pytesseract.tesseract_cmd =r"/usr/local/Cellar/tesseract/5.3.4/bin/tesseract"```, and replace the string with your path to the executable. 
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
The ```demo.ipynb``` is a jupyter notebook with an example of how to use this code. The two main functions in this repo are ```extract_all_addresses(input_csv_path, file_dir, output_csv_path)``` and ```extract_all_init_demands(input_csv_path, file_dir, output_csv_path)```. To call either of them, simply pass 
* the path to the input csv. 
  * This is a file where each row represents a separate legal file in a particular case (e.g. civil case cover sheet, complaint). It should contain two columns of interest: 
    * `case_number`, containing the case identifier
    * `Document`, containing the description of the document type (e.g. "Civil Case Cover Sheet", "Complaint")
* the path to the folder containing all the actual pdf documents
* the path of a new output csv.

The functions will iterate over all the case numbers, attempt to extract the desired information, and save the results to a new csv. If it could not extract the information, it will leave the entry blank. It will also log whether or not the automated extraction passed or failed in the Notes section of the csv, along with an error message if there is one. 

## How it works: 
### Initial demand 
The process to extract the initial demand is fairly straightforward. There appear to be a few general versions of the Complaint file:
* page 1:
    * DEMAND: $XXXXX.XX
    * DEMAND AMOUNT: $XXXXX.XX
    * AMOUNT OF DEMAND: $XX,XXX.XX
    * AMOUNT DEMANDED: $XX,XXX.XX
    * Demand is for $XXXXX.XX
    * PRAYER AMOUNT: $XXXXX.XX
    * PRAYER AMT: $XXXXX.XX
    * LIMITED CIVIL: $XXXXX.XX
* page 2:
    * 10\. Plaintiff prays for judgment for costs of suit; for such
      relief as is fair, just, and equitable; and for a. damages of:
      $XXXXX.XX
* page 3:
    * WHEREFORE, as to all Causes of Action, Plaintiff prays for
      judgment against Defendant, including but not limited to, the
      amounts as follows: For damages of $XXXXX.XX;
      
(I have not been able to discern a pattern between cases that have different forms.) The extraction process simply runs OCR over the first three pages of the Complaint, then checks to see if it can detect each of these patterns using regular expressions (regex); if so, it extracts the string, cleans it up, and returns the float value. 

Note that the ~10% of cases for which the automated extraction fail are not necessarily random – for example, several of them are because there are multiple defendants and the initial demand is not listed on the first 3 pages in the expected format. Thus, we may be systematically losing information if we exclude the cases that we can't automatically extract the demand from. 

### Address 
Automating address extraction is much more complicated. It isn't as easy as running OCR over the entire page containing the address because the page isn't single column; the address fields are located in their own boxes. The address text often gets mixed together with other text from the page, making it very difficult to clean, create a regex pattern, or logic through. 

There are lots of workarounds, but the more successful ones probably all require some amount of computer vision. In the current address extraction algorithm, we use some very basic box detection which someone else has kindly provided open-source (that algorithm is still extremely sensitive and often fails to detect boxes). By specifying the expected sizes for the street, city, state, and zip code boxes, we are able to detect the respective boxes on the form and extract crops of the image containing only that text. Then, when we run OCR over the inividual address field boxes, we are able to extract the address field text pretty cleanly. 


## Alternative strategies and things to improve: 
* Test accuracy of OCR against the hand-collected data. Even though we are able to extract information for 65-70% of cases, this figure only represents the files where the code didn't crash, and does not represent the accuracy of the extracted information. 
* Improve running time. Currently the address extraction can process around 350 cases an hour. For 500,000 cases, it would take around 1,400 hours, or 60 days 😬. 
 * downsize the images, when feasible (probably want to keep it around 300 DPI for the OCR, though) 
 * more aggressive cropping to reduce image sizes 
 * improve the box detection algorithm so we don't have to rotate it by 0.1 degrees like 20 times
 * Run things on a GPU
 * For initial demand, check one page at a time, and only process later pages if we can't find demand on an earlier page (instead of running OCR on all 3 pages at the start) 
* Add intermediate saving steps. Useful in case you want to pause midway through or when accidents happen. 

### For initial demand extraction
* I suspect that a lot of the cases where the algorithm fails to extract the initial demand is because the OCR has some typos and fails to match the regex pattern. We should double check this, and if this is the case, figure out a way to look for strings within a small edit distance.

### For address extraction
* We are still sometimes getting minor OCR errors sneaking in, like diacritics and random punctuation.
  * It should be fairly easy to filter out the random punctuation
  * removing diacritics/restricting the characters to the latin alphabet is a bit of a problem because apparently the tesseract character whitelist only works for the legacy mode, not the newer neural net/LSTM mode, which is what we use in this repo (https://stackoverflow.com/a/49030935/10536083). We could try comparing the performance of the neural net vs legacy-with-whitelist to see what is better. 
* Sometimes the typos are just so bad as to be unrecognizable (e.g. OI instead of CITY) but the actual name of the city is correctly parsed by the OCR. so if we check for the word CITY instead of the city name directly (which we currently do), we're missing out on some files that we could successfully parse. an alternative is to check for the presence of a valid LA city name, which we have a list of. (we would probably also want to check for typos with small edit distance)
* The box detection algorithm is very sensitive to rotation and very finicky to use. we could fork that repo and try to improve it (doable but would take a while)
* Sometimes the box detection algorithm detects the wrong box (e.g. multiple times we have detected the ZIP CODE box when we really wanted the CITY box).
 * An alternative strategy is to first do a pass of OCR over the entire page to detect the presence of keywords and then get the bounding box of those key words (which is a function in tesseract). Then, given those reference coordinates, we could restrict box detection to a much smaller section of the page, which should reduce the chances of detecting the wrong box (and possibly of failing to detecting the box at all)
 * Another alternative strategy is to find all boxes on a page, then check each one for the text ADDRESS, CITY, STATE, ZIP CODE, etc. However, because the box detection code is so finicky and unreliable, it sometimes sometimes can't find two boxes of the same size on the same page that it should be able to find, thus we might not actually extract all the boxes on the page. Again, this would require improving the actual box detection algorithm. 


<!-- ## Background research of possible existing solutions: -->
