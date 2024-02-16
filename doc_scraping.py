# imports
# NOTE: a couple local installations are required: tesseract and poppler
# for poppler installation instructions, see here (steps differ by operating system): https://pdf2image.readthedocs.io/en/latest/installation.html
# for tesseract installation, see here: https://tesseract-ocr.github.io/tessdoc/Installation.html
import numpy as np
import pandas as pd
import os
import re
from pdf2image import convert_from_path
from scipy import ndimage
import pytesseract
import matplotlib.pyplot as plt
from boxdetect import config
from boxdetect.pipelines import get_boxes

# add the tesseract executable to your PATH, or run the following command
# pytesseract.pytesseract.tesseract_cmd =r"/usr/local/Cellar/tesseract/5.3.4/bin/tesseract"
# you can test if tesseract is installed by calling `tesseract` in your command line (without the backticks)

DPI = 300


def get_file(case_number, file_type, file_dir):
    """ Retrieve the path to a desired file given the file description and case 
        number. 

        Args:
            case_number (str): case identifier, alphanumeric
            file_type (str): the name of the legal file, 
                e.g. "complaint", "civil_case_cover_sheet"
            file_dir (str): path to the folder containg all the scanned legal 
                documents
    
        Returns: list of strings containing the file paths
    """
    case_number = case_number.lower()
    file_type = file_type.lower()
    possible_files = []
    for f in os.listdir(file_dir):
        if f.lower().startswith(case_number) and file_type in f.lower():
            if file_type != 'complaint' or (file_type == 'complaint'
                                            and 'summons_on_complaint'
                                            not in f.lower()):
                possible_files.append(os.path.join(file_dir, f))
    return possible_files


def cover_sheet_last_page_image(case_number, file_dir):
    """ Gets the last page of the cover sheet, assuming it should have at least 
        6 pages. 
        
        Note that the address is almost always on the 6th page (occasionally 
        there is a 7th page that it will be found on, sometimes there are fewer 
        pages and missing an address)
    
        Args:
            case_number (str): case identifier, alphanumeric
            file_dir (str): path to the folder containg all the scanned legal 
                documents
    
        Returns: PIL image
    """
    # retrieve civil case cover file sheet
    fpaths = get_file(case_number, 'civil_case_cover_sheet', file_dir)
    if len(fpaths) == 0:
        raise Exception(
            f'could not find civil case cover sheet for case {case_number}')
    elif len(fpaths) > 1:
        raise Exception(
            f'found {len(fpaths)} civil case cover sheets for case {case_number}'
        )

    # convert pdf to image
    # the address is almost always on the last (6th) sheet
    images = convert_from_path(fpaths[0], dpi=DPI, first_page=6)

    # not all civil case cover sheets have 6 pages, which means they might be missing the address in that document, so test for that
    if len(images) == 0:
        raise Exception(
            'civil case cover sheet does not have 6 pages as expected; likely missing address'
        )
    last_page_image = images[-1]

    return last_page_image


def address_autocrop(last_page_image):
    """ Automatically detect the boxes for each field in the address page of 
        the civil case cover sheet. 

        last_page_image: image object containing the last page of the civil 
            case cover sheet. can be various types, such as PIL image.
    
        Returns: tuple of 4 np arrays containing the images of the cropped 
            boxes for the address, city, state, and zip code
    """
    last_page_image = np.array(last_page_image)
    # crop the last page to the top half since the file is large and makes transformations slow
    # also crop a bit off the top since there is a box that sometimes gets confused for the address box
    top_crop = 300
    half_height = int(0.5 * last_page_image.shape[0])
    last_page_image = last_page_image[top_crop:half_height, :]

    ## create the pipeline configs for the box detectors for the street address box, city box, and state/zip boxes ##

    ADDRESS_CFG = config.PipelinesConfig()
    # important to adjust these values to match the size of boxes on your image
    ADDRESS_CFG.width_range = (800, 1500)
    ADDRESS_CFG.height_range = (150, 500)
    # there are some as short as 97, but reducing the range this much makes it detect the wrong box more often than not, I think

    # the more scaling factors the more accurate the results but also it takes more time to processing
    # too small scaling factor may cause false positives
    # too big scaling factor will take a lot of processing time
    # resizes image based on scaling factor
    ADDRESS_CFG.scaling_factors = [0.7]
    # w/h ratio range for boxes/rectangles filtering
    ADDRESS_CFG.wh_ratio_range = (0.5, 12)
    # group_size_range starting from 2 will skip all the groups
    # with a single box detected inside (like checkboxes)
    ADDRESS_CFG.group_size_range = (2, 100)
    # num of iterations when running dilation tranformation (to engance the image)
    ADDRESS_CFG.dilation_iterations = 5

    CITY_CFG = config.PipelinesConfig()
    CITY_CFG.width_range = (250, 850)
    CITY_CFG.height_range = (90, 210)
    CITY_CFG.scaling_factors = [0.7]
    CITY_CFG.wh_ratio_range = (1.5, 7.0)
    CITY_CFG.group_size_range = (2, 100)
    CITY_CFG.dilation_iterations = 2

    STATEZIP_CFG = config.PipelinesConfig()
    STATEZIP_CFG.width_range = (150, 450)
    STATEZIP_CFG.height_range = (90, 210)
    STATEZIP_CFG.scaling_factors = [0.7]
    STATEZIP_CFG.wh_ratio_range = (0.6, 5)
    STATEZIP_CFG.group_size_range = (2, 100)
    STATEZIP_CFG.dilation_iterations = 2

    ## try to locate the field boxes now ##

    # for the street address, crop to the right side of the page so we remove the REASON box
    right_crop = 1000
    last_page_image_right = last_page_image[:, right_crop:]
    street_address_bbox_candidates, _, _, _ = get_boxes(last_page_image_right,
                                                        cfg=ADDRESS_CFG,
                                                        plot=False)
    # if no boxes detected, its likely because the scan was rotated. the box detection algorithm is super sensitive to angle (i.e. within 0.2 degrees), so we rotate the image +- 0.1 degrees until we can detect boxes
    max_rot = 2
    if len(street_address_bbox_candidates) == 0:
        deg = 0
        print(
            'could not find street address box, testing different image rotations now'
        )
        while len(street_address_bbox_candidates) == 0 and abs(deg) < max_rot:
            # rotate +0.1 deg
            deg = abs(deg) + 0.1
            rotated = ndimage.rotate(last_page_image_right, deg, reshape=False)
            # try box detection again
            street_address_bbox_candidates, _, _, _ = get_boxes(
                rotated, cfg=ADDRESS_CFG, plot=False)
            if len(street_address_bbox_candidates) != 0:
                print(f'box detected successfully at {deg} degree rotation')
                break

            # if that doesn't work, rotate in the opposite direction
            deg = -deg
            rotated = ndimage.rotate(last_page_image_right, deg, reshape=False)
            # try box detection again
            street_address_bbox_candidates, _, _, _ = get_boxes(
                rotated, cfg=ADDRESS_CFG, plot=False)
            if len(street_address_bbox_candidates) != 0:
                print(f'box detected successfully at {deg} degree rotation')
                break

    # verify if the street_address_bbox_candidates has valid boxes now or if its still empty
    if len(street_address_bbox_candidates) == 0:
        raise Exception(
            f'could not detect address box within +-{max_rot} degrees of rotation'
        )

    city_bbox_candidates, _, _, _ = get_boxes(last_page_image,
                                              cfg=CITY_CFG,
                                              plot=False)
    # we have to repeat the rotation stuff because it's possible that city/state/zip will be recognized at different rotations from street address
    if len(city_bbox_candidates) == 0:
        deg = 0
        print('could not find city box, testing different image rotations now')
        while len(city_bbox_candidates) == 0 and abs(deg) < max_rot:
            # rotate +0.1 deg
            deg = abs(deg) + 0.1
            rotated = ndimage.rotate(last_page_image, deg, reshape=False)
            # try box detection again
            city_bbox_candidates, _, _, _ = get_boxes(rotated,
                                                      cfg=CITY_CFG,
                                                      plot=False)
            if len(city_bbox_candidates) != 0:
                print(f'box detected successfully at {deg} degree rotation')
                break

            # if that doesn't work, rotate in the opposite direction
            deg = -deg
            rotated = ndimage.rotate(last_page_image, deg, reshape=False)
            # try box detection again
            city_bbox_candidates, _, _, _ = get_boxes(rotated,
                                                      cfg=CITY_CFG,
                                                      plot=False)
            if len(city_bbox_candidates) != 0:
                print(f'box detected successfully at {deg} degree rotation')
                break

    # verify if the city_bbox_candidates has valid boxes or if its empty
    if len(city_bbox_candidates) == 0:
        raise Exception(
            f'could not detect address box within +-{max_rot} degrees of rotation'
        )

    state_zip_bbox_candidates, _, _, _ = get_boxes(last_page_image,
                                                   cfg=STATEZIP_CFG,
                                                   plot=False)
    if len(state_zip_bbox_candidates) < 2:
        deg = 0
        print(
            'could not find 2 state/zip boxes, testing different image rotations now'
        )
        while len(state_zip_bbox_candidates) < 2 and abs(deg) < max_rot:
            # rotate +0.1 deg
            deg = abs(deg) + 0.1
            rotated = ndimage.rotate(last_page_image, deg, reshape=False)
            # try box detection again
            state_zip_bbox_candidates, _, _, _ = get_boxes(rotated,
                                                           cfg=STATEZIP_CFG,
                                                           plot=False)
            if len(state_zip_bbox_candidates) >= 2:
                print(f'boxes detected successfully at {deg} degree rotation')
                break

            # if that doesn't work, rotate in the opposite direction
            deg = -deg
            rotated = ndimage.rotate(last_page_image, deg, reshape=False)
            # try box detection again
            state_zip_bbox_candidates, _, _, _ = get_boxes(rotated,
                                                           cfg=STATEZIP_CFG,
                                                           plot=False)
            if len(state_zip_bbox_candidates) >= 2:
                print(f'box detected successfully at {deg} degree rotation')
                break
    # verify if the state_zip_bbox_candidates has valid boxes or if its empty
    if len(state_zip_bbox_candidates) < 2:
        raise Exception(
            f'fewer than 2 boxes found for State and Zip Code fields after rotation corrections of +-{max_rot} degrees; aborting'
        )

    # the format of the bbox coordinates is [left, top, width, height]
    # extract bounding box for street address
    if street_address_bbox_candidates.shape[0] == 1:
        # only one street address box found, use it
        street_address_bbox = street_address_bbox_candidates[0]
    else:
        # TODO just check the boxes for text matching ADDRESS – and do that earlier when rotating degrees in case we detect the empty box under address but not the actual address box itself (happens for some cases)
        # take the rightmost box, since we likely also got the Reason box
        street_address_bbox = street_address_bbox_candidates[0]
        for i in range(1, street_address_bbox_candidates.shape[0]):
            left_coord = street_address_bbox_candidates[i][0]
            if left_coord > street_address_bbox[0]:
                street_address_bbox = street_address_bbox_candidates[i]

    # extract bounding box for city
    if city_bbox_candidates.shape[0] == 1:
        # only one city box found, use it
        city_bbox = city_bbox_candidates[0]
    else:
        # take the leftmost box, since we likely also got the State box
        city_bbox = city_bbox_candidates[0]
        for i in range(1, city_bbox_candidates.shape[0]):
            left_coord = city_bbox_candidates[i][0]
            if left_coord < city_bbox[0]:
                city_bbox = city_bbox_candidates[i]

    # extract bounding box for state and zip
    if state_zip_bbox_candidates.shape[0] > 2:
        # more than two boxes found
        # filter by general horizontal position and remove the boxes we think are incorrect
        # horizontally the top should be somewhere below 750px and the bottom should be somewhere above 1600px
        state_zip_bbox_candidates_orig = state_zip_bbox_candidates
        state_zip_bbox_candidates = []
        for bbox in state_zip_bbox_candidates_orig:
            top = bbox[1]
            bottom = bbox[1] + bbox[3]
            if top > 650 and bottom < 1700:
                state_zip_bbox_candidates.append(bbox)
        state_zip_bbox_candidates = np.array(state_zip_bbox_candidates)

        # by the end of this filtering process we should hopefully have exactly 2 boxes left
        if state_zip_bbox_candidates.shape[0] != 2:
            raise Exception(
                'after filtering out possible wrong state/zip boxes, fewer than 2 boxes remaining; aborting'
            )

    # we should be left with exactly two boxes now
    # should be state and zip, state is on the left
    if state_zip_bbox_candidates[0][0] < state_zip_bbox_candidates[1][0]:
        state_bbox = state_zip_bbox_candidates[0]
        zip_bbox = state_zip_bbox_candidates[1]
    else:
        state_bbox = state_zip_bbox_candidates[1]
        zip_bbox = state_zip_bbox_candidates[0]

    # make the crops
    # and add a little bit of padding around the crops in case the auto bbox was too tight (the OCR doesn't like very tight crops)
    address_crop = last_page_image_right[street_address_bbox[1] -
                                         30:street_address_bbox[1] +
                                         street_address_bbox[3] + 30,
                                         street_address_bbox[0] -
                                         30:street_address_bbox[0] +
                                         street_address_bbox[2] + 30]
    city_crop = last_page_image[city_bbox[1] - 10:city_bbox[1] + city_bbox[3] +
                                10, city_bbox[0] - 50:city_bbox[0] +
                                city_bbox[2] + 10]
    state_crop = last_page_image[state_bbox[1] - 10:state_bbox[1] +
                                 state_bbox[3] + 10, state_bbox[0] -
                                 10:state_bbox[0] + state_bbox[2] + 10]
    # add extra padding to the bottom of the zip code bbox since sometimes the zip code is 2 lines long and exceeds the bounds
    zip_crop = last_page_image[zip_bbox[1] - 10:zip_bbox[1] + zip_bbox[3] + 40,
                               zip_bbox[0] - 10:zip_bbox[0] + zip_bbox[2] + 10]

    return address_crop, city_crop, state_crop, zip_crop


def address_from_crops(address_crop,
                       city_crop,
                       state_crop,
                       zip_crop,
                       verbose=True):
    """ Extract text using OCR from the cropped boxes around each address field 
        on the civil case cover sheet. 

        <address/city/state/zip>_crop: image cropped around the respective 
            address field
        verbose (bool): whether or not to print the retrieved address fields
    
        Returns: tuple of strings containing street address, city, state, and zip code, if extraction was successful 
    """
    # run OCR on the crops
    tesseract_config = f"--oem 1 --dpi {DPI}"
    # NOTE the tesseract characte whitelist only works for the legacy mode, not the newer neural net/LSTM mode, which apparently doesn't respect the whitelist - so we can't really force it to exclude weird punctuation or diacritics
    #  (https://stackoverflow.com/a/49030935/10536083)
    streetaddress_texts = pytesseract.image_to_string(address_crop,
                                                      config=tesseract_config)
    city_texts = pytesseract.image_to_string(city_crop,
                                             config=tesseract_config)
    state_texts = pytesseract.image_to_string(state_crop,
                                              config=tesseract_config)
    zip_texts = pytesseract.image_to_string(zip_crop, config=tesseract_config)

    #### extract the address info from the text blocks ####
    # drop the text containing "CITY"/"ADDRESS"/etc, remove punctuation, strip whitespace
    # TODO filter out artifacts of OCR, like random characters like -_:;'" (note we can't filter out periods or commas because those could be valid instances in the address) (usually commas for separating street/apt, and periods following the street abbreviation)

    ## extract street address ##
    # strip whitespace so that it should start with 'ADDRESS'
    streetaddress_texts = streetaddress_texts.strip()
    address_start_idx = streetaddress_texts.upper().find('ADDRESS')
    # also check for typos of ADDRESS
    # TODO create a more flexible way of checking for typos using edit distance
    if address_start_idx == -1:
        address_start_idx = streetaddress_texts.upper().find('ADORESS')
    if address_start_idx == -1:
        address_start_idx = streetaddress_texts.upper().find('AOORESS')
    if address_start_idx == -1:
        address_start_idx = streetaddress_texts.upper().find('AODRESS')
    assert address_start_idx != -1, f'the street address block should start with ADDRESS but could not find that word in: {streetaddress_texts}'

    # remove the starting text corresponding to 'ADDRESS'
    streetaddress = streetaddress_texts[address_start_idx + len('ADDRESS'):]
    # remove the semicolon, or similar punctuation that the OCR might've misinterpreted for the semicolon. sometimes it doesn't catch the semicolon so we also need to check for that
    streetaddress = streetaddress.strip()
    if streetaddress[0] in [':', ';', ',', '.', '\'']:
        streetaddress = streetaddress[1:]
        # strip any remaining whitespace
        streetaddress = streetaddress.strip()
    # filter out defendant name (assuming name will only ever contain alphabetic characters, and that the street address always starts with numbers for the building)
    name_filter = "[a-zA-Z\s]*(\d.*)"
    streetaddress = re.findall(name_filter, streetaddress)[0]
    streetaddress = streetaddress.strip()

    ## extract city ##
    # strip whitespace so that it should start with 'CITY'
    city_texts = city_texts.strip()
    city_start_idx = city_texts.upper().find('CITY')
    # also check for typos of CITY (I've seen CHY, OI)
    if city_start_idx == -1:
        city_start_idx = city_texts.upper().find('CHY')
    assert city_start_idx != -1, f'the city block should start with CITY but could not find that word in: {city_texts}'
    # remove the starting text corresponding to 'CITY'
    city = city_texts[city_start_idx + len('CITY'):]
    city = city.strip()
    # remove the semicolon, or similar punctuation that the OCR might've misinterpreted for the semicolon. sometimes it doesn't catch the semicolon so we also need to check for that
    if city[0] in [':', ';', ',', '.', '\'']:
        city = city[1:]
        # strip any remaining whitespace
        city = city.strip()

    ## extract state ##
    # strip whitespace so that it should start with 'STATE'
    state_texts = state_texts.strip()
    state_start_idx = state_texts.upper().find('STATE')
    assert state_start_idx != -1, f'the state block should start with STATE but could not find that word in: {state_texts}'
    # remove the starting text corresponding to 'STATE'
    state = state_texts[state_start_idx + len('STATE'):]
    state = state.strip()
    # remove the semicolon, or similar punctuation that the OCR might've misinterpreted for the semicolon. sometimes it doesn't catch the semicolon so we also need to check for that
    if state[0] in [':', ';', ',', '.', '\'']:
        state = state[1:]
        # strip any remaining whitespace
        state = state.strip()

    ## extract zip code ##
    # strip whitespace so that it should start with 'ZIP'
    zip_texts = zip_texts.strip()
    zip_start_idx = zip_texts.upper().find('ZIP')
    assert zip_start_idx != -1, f'the zip block should start with ZIP but could not find that word in: {zip_texts}'
    # remove the starting text corresponding to 'ZIP'
    zip_texts = zip_texts[zip_start_idx + len('ZIP'):]

    # it's posible the space between ZIP and CODE didn't get recognized properly, so we remove the word CODE in a separate step
    # strip again to get rid of whitespace before CODE
    zip_texts = zip_texts.strip()
    code_start_idx = zip_texts.upper().find('CODE')
    assert code_start_idx != -1, f'the zip block should have CODE as the second word but could not find that word in: {zip_texts}'
    # remove the starting text corresponding to 'CODE'
    zip_texts = zip_texts[code_start_idx + len('CODE'):]
    zip_texts = zip_texts.strip()

    # remove the semicolon, or similar punctuation that the OCR might've misinterpreted for the semicolon. sometimes it doesn't catch the semicolon so we also need to check for that
    if zip_texts[0] in [':', ';', ',', '.', '\'']:
        zip_texts = zip_texts[1:]
    # strip any remaining whitespace
    zip_texts = zip_texts.strip()

    # regex to parse the zip code, because sometimes the second section (+4 digits) of the zip code gets botched by OCR, or is not included
    zipcode_5 = zip_texts[:5]
    # ^ this is the first part of the zip code with 5 digits, which they should all have
    # TODO check this since it seemed to have wrongly let through a 4 digit code
    assert zipcode_5.isdigit(
    ), f"the first 5 characters of the zip code should be all digits, but instead found: {zipcode_5}"

    # then check if we have a valid second part of the zip code with 4 digits (this is optional so if we can't find a valid version, we just drop it)
    # sometimes we don't have anything, so we have to catch that case too
    zip_texts = zip_texts[5:].strip()
    zipcode = zipcode_5
    if len(zip_texts) > 0:
        # drop the hyphen if we find one (sometimes there isn't)
        if zip_texts[0] in ['-', '_', '–']:
            zip_texts = zip_texts[1:]
        # then get the next 4 characters and check if they are digitis
        zip_texts = zip_texts.strip()
        zipcode_4 = zip_texts[:4]
        if zipcode_4.isdigit():
            zipcode = zipcode_5 + '-' + zipcode_4
    # if not, we just ignore, there was probably an OCR error and the zip+4 isn't required to geolocate the address

    # display the original scans and extracted address in case we want to verify results
    if verbose:
        print("extracted address")
        print("street address: ", streetaddress)
        print("city: ", city)
        print("state: ", state)
        print("zipcode: ", zipcode)

    return streetaddress, city, state, zipcode


# TODO replace asserts with if raises


def extract_address(case_number,
                    file_dir,
                    view_scans=False,
                    print_address=True):
    """ Extract the address from the civil case cover sheet for a single case. 
    
        Args:
            case_number (str): case identifier, alphanumeric
            file_dir (str): path to the folder containg all the scanned legal 
                documents
            view_scans (bool): whether or not to plot the address image crops 
                (for debugging/verification)
            print_address (bool): whether or not to print the extracted address 
                (for debugging/verification)
    
        Returns: tuple of strings containing street address, city, state, and 
            zip code, if extraction was successful 
    """
    last_page_image = cover_sheet_last_page_image(case_number, file_dir)

    address_crop, city_crop, state_crop, zip_crop = address_autocrop(
        last_page_image)

    # display the cropped scans in case we want to verify results
    if view_scans:
        print("address autocrops")
        fig, axs = plt.subplots(1, 4, figsize=(15, 2))
        axs = axs.flatten()
        axs[0].imshow(address_crop)
        axs[1].imshow(city_crop)
        axs[2].imshow(state_crop)
        axs[3].imshow(zip_crop)
        axs[0].set_title('Street address crop')
        axs[1].set_title('City crop')
        axs[2].set_title('State crop')
        axs[3].set_title('Zip code crop')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')

    streetaddress, city, state, zipcode = address_from_crops(address_crop,
                                                             city_crop,
                                                             state_crop,
                                                             zip_crop,
                                                             verbose=False)

    # display the extracted address in case we want to verify results
    if print_address:
        print("extracted address")
        print("street address: ", streetaddress)
        print("city: ", city)
        print("state: ", state)
        print("zipcode: ", zipcode)

    return streetaddress, city, state, zipcode


def extract_init_demand(case_number, file_dir, verbose=False):
    """ Extract the initial demand from the complaint for a single case. 
    
        Args:
            case_number (str): case identifier, alphanumeric
            file_dir (str): path to the folder containg all the scanned legal 
                documents
            verbose (bool): whether or not to print extra details (for 
                debugging/verification)
    
        Returns: float value of the initial demand, as well as page the value was found on (for debugging purposes)
    """
    fpaths = get_file(case_number, 'complaint', file_dir)
    if len(fpaths) == 0:
        raise Exception(f'could not complaint for case {case_number}')
    elif len(fpaths) > 1:
        raise Exception(
            f'found {len(fpaths)} complaints for case {case_number}')

    # initial demand should be on the first or second page of the complaint
    images = convert_from_path(fpaths[0], dpi=DPI, last_page=2)

    # verify that we have 2 pages from the doc
    if len(images) < 2:
        raise Exception(
            'complaint has fewer than 2 pages, may be missing initial demand')
    # TODO we should still try checking first page even if we don't have second page. but tbh i think it'll be very unlikely we have less than 2 pages

    # run OCR
    tesseract_config = f"--oem 1 --dpi {DPI}"
    # NOTE the tesseract characte whitelist only works for the legacy mode, not the newer neural net/LSTM mode, which apparently doesn't respect the whitelist
    first_page_text = pytesseract.image_to_string(images[0],
                                                  config=tesseract_config)
    second_page_text = pytesseract.image_to_string(images[1],
                                                   config=tesseract_config)

    # check which version of the complaint form is used by searching for different variations of the text:
    #     * DEMAND: $XXXXX.XX
    #     * PRAYER AMOUNT: $XXXXX.XX
    #     * 10. Plaintiff prays for judgment for costs of suit; for such relief
    #       as is fair, just, and equitable; and for a. damages of: $XXXXX.XX
    # NOTE: we assume between 3-5 digits on the left integer side of the decimal (since should be <$25,000 and plaintiffs probably won't sue if it's <$100)
    # use regex to extract monetary amount
    prayer_amount_regex = "[pP][rR][aA][yY][eE][rR]\s*[aA][mM][oO][uU][nN][tT]\s*[:;,.-]?\s*[$Ss]\s*(\d{0,2}[,.]?\d{0,3}[.,]?\d{2})"
    demand_regex = "[dD][eE][mM][aA][nN][dD]\s*[:;,.-]?\s\s*[$Ss]\s*(\d{0,2}[,.]?\d{0,3}[.,]?\d{2})"
    plaintiff_prays_regex = "damages\s*of\s*[:;,.-]?\s*[$Ss]\s*(\d{0,2}[,.]?\d{0,3}[.,]?\d{2})"
    # TODO figure out a way to make these more robust to OCR typos
    # note sometimes the decimal point in the monetary value doesn't get detected by OCR, hence why we make them optional in the regex pattern
    prayer_amount_results = re.findall(prayer_amount_regex, first_page_text)
    demand_results = re.findall(demand_regex, first_page_text)
    plaintiff_prays_results = re.findall(plaintiff_prays_regex,
                                         second_page_text)

    # convert to value
    demands_found = prayer_amount_results + demand_results + plaintiff_prays_results
    if len(demands_found) == 0:
        raise Exception(
            'could not find initial demand on first or second page; aborting')
    if len(demands_found) > 1:
        print('found multiple instances of initial demand, which is weird')
        # if we somehow find multiple demands, just take the first one for now
        # TODO add better handling later
    if len(prayer_amount_results) > 0:
        found_on = 'page 1 (as PRAYER AMOUNT)'
    if len(demand_results) > 0:
        found_on = 'page 1 (as DEMAND)'
    if len(plaintiff_prays_results) > 0:
        found_on = 'page 2 (section 10. Plaintiff prays. . . for damages of)'

    if verbose:
        print(f'found initial demand on {found_on}')

    # convert to float
    # note we need to remove any commas before converting to float. however, it's possible that the decimal point will get interpreted as a comma, or won't get detected at all by the OCR. thus, let's just remove all punctuation, and then re insert it back in, assuming the demands always have a decimal value (TODO we should verify this assumption)
    init_demand_str = demands_found[0]
    init_demand_str_no_punc = re.sub('[.,]', '', init_demand_str)
    init_demand = float(init_demand_str_no_punc) / 100

    if init_demand > 25_000:
        raise Warning(
            'detected initial demand is greater than $25,000, is this expected?'
        )

    return init_demand, found_on


def extract_all_addresses(input_csv_path, file_dir, output_csv_path):
    """ Extract the address from the civil case cover sheet for all cases in a 
        csv.
    
        Args:
            input_csv_path (str): path to csv containing case entries (with all 
            the assigned case numbers, file names, etc but presumably no 
            addresses)
            file_dir (str): path to the folder containg all the scanned legal 
                documents
            output_csv_path (str): path to which updated csv with addresses 
                will be saved
    
        Output: saves new csv. 
    """
    # load the csv as a pandas dataframe
    df = pd.read_csv(input_csv_path)

    # iterate over case number
    error_count = 0
    for i, case_id in enumerate(df['case_number'].unique()):
        print(i + 1, ":", case_id)
        try:
            streetaddress, city, state, zipcode = extract_address(
                case_id, file_dir, view_scans=False, print_address=True)
            # add to dataframe
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Civil Case Cover Sheet'),
                   'address'] = (streetaddress + ", " + city + " " + state +
                                 " " + zipcode)
            # add additional note columns so we know if it was automated/if
            # there were any issues
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Civil Case Cover Sheet'),
                   'automated address'] = 'passed'
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Civil Case Cover Sheet'),
                   'automated address error'] = ''
            print()
        except Exception as e:
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Civil Case Cover Sheet'),
                   'automated address'] = 'failed'
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Civil Case Cover Sheet'),
                   'automated address error'] = e
            error_count += 1
            print(e)
            print(f'errors: {error_count}/{i+1}')
            print()

    # save to a new csv
    df.to_csv(output_csv_path)


def extract_all_init_demands(input_csv_path, file_dir, output_csv_path):
    """ Extract the initial demand from the complaint for all cases in a csv.
    
        Args:
            input_csv_path (str): path to csv containing case entries (with all 
            the assigned case numbers, file names, etc but presumably no 
            addresses)
            file_dir (str): path to the folder containg all the scanned legal 
                documents
            output_csv_path (str): path to which updated csv with addresses 
                will be saved
    
        Output: saves new csv. 
    """
    # load the csv as a pandas dataframe
    df = pd.read_csv(input_csv_path)

    # iterate over case number
    error_count = 0
    for i, case_id in enumerate(df['case_number'].unique()):
        print(i + 1, ":", case_id)
        try:
            init_demand, found_on = extract_init_demand(case_id, file_dir)
            print('init_demand', init_demand)

            # add to dataframe
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Complaint'),
                   'initial demand amount'] = init_demand

            # add additional note columns so we know if it was automated/if
            # there were any issues
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Complaint'),
                   'automated initial demand'] = 'passed'
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Complaint'),
                   'automated initial demand error'] = ''
            print()

        except Exception as e:
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Complaint'),
                   'automated initial demand'] = 'failed'
            df.loc[(df['case_number'] == case_id) &
                   (df['Document'] == 'Complaint'),
                   'automated initial demand error'] = e
            error_count += 1
            print(e)
            print(f'errors: {error_count}/{i+1}')
            print()

    # save to a new csv
    df.to_csv(output_csv_path)
