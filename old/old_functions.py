from functions import *

def process_images_and_data(img_T_replaced, img_no_equipment, img, ocr_results_with_rotated_text,
                            re_lines, re_equipment, erosion, shrink_factor, lower_thresh, line_box_expand, services,
                            equipment_boxes, equipment_box_expand, include_inside, img_scale):
    # Process line images
    processed_img_lines = process_image(img_T_replaced, erosion, shrink_factor, lower_thresh=lower_thresh)
    lines = get_text_in_ocr_results(re_lines, ocr_results_with_rotated_text)
    line_colors = return_hash_colors(lines)

    line_img = generate_painted_img(processed_img_lines, lines, line_colors, shrink_factor, expand_scale=line_box_expand)


    service_colors = return_service_colors(services, line_img, img_scale)
    processed_service_img = process_image(img_no_equipment, erosion, shrink_factor, lower_thresh=lower_thresh)

    #if we could just generate services_in and services_out as text lists with indexes equal to lines that'd be great

    # Process service in lines. idea is to return a subset of lines and colors, so we can repaint the image
    # services_in = get_services(lines, line_colors, services, service_colors, 'service_in')

    services_in, services_in_colors, services_in_txt = return_serviced_lines(lines, line_colors, services, service_colors, 'service_in')
    service_in_img = generate_painted_img(processed_service_img, services_in, services_in_colors, shrink_factor,
                                          expand_scale=line_box_expand)

    # Process service out lines where each index corresponds ot a service out for that line/line_color index
    services_out, services_out_colors, services_out_txt = return_serviced_lines(lines, line_colors, services, service_colors, 'service_out')
    service_out_img = generate_painted_img(processed_service_img, services_out, services_out_colors, shrink_factor,
                                           expand_scale=line_box_expand)

    # Process equipment images
    processed_img = process_image(img, erosion, shrink_factor, lower_thresh=lower_thresh)
    equipments, equipment_img = generate_equipment(processed_img, ocr_results_with_rotated_text, re_equipment,
                                                   shrink_factor, equipment_boxes, equipment_box_expand,
                                                   include_inside)
    # cv2.imwrite(os.path.join(results_folder, 'equipment_img.png'), equipment_img)





    services_in = list(zip(services_in_colors, services_in_txt))
    services_out = list(zip(services_out_colors, services_out_txt))


    return line_img, equipment_img, service_in_img, service_out_img, lines, line_colors, equipments, services_in, services_out


def return_service_colors(services, line_img, img_scale):
    # services = [[text, label, box]...]
    service_colors = []
    for service in services:
        text, lb, box = service
        service_color = region_mode_color(box, line_img, img_scale)
        service_colors.append(service_color)

    return service_colors


def figure_out_if_instrument_has_sis(data, img, shrink_factor):
    # box, tag, tag_no, label, line_id, comment, valve_type, valve_size, inst_alarm, sis = data
    # page, pid, box, tag, tag_no, label, line_id, comment, valve_type, valve_size, inst_alarm
    # print(data)

    for x in data:
        if x['tag'] == 'SIS':
            tag_no = x['tag_no']
            print(f'found a sis {tag_no}')
            start = get_box_center(x['box'])
            img_copy = copy.copy(img)
            flood_fill_till_inst_reached(start, tag_no, data, img_copy, shrink_factor)

def generate_equipment(processed_img, ocr_results, re_pattern, shrink_factor,
                       detecto_equipment_boxes, equipment_box_expand, include_inside):
    #generate_equipment(line_img, shrink_factor, re_pattern, detecto_equipment_boxes, equipment_box_expand, include_inside, ocr_results):
    #generate_equipment(processed_img, ocr_results_with_rotated_text, re_equipment, shrink_factor,
    #                   equipment_boxes, equipment_box_expand, include_inside)
    # img=img_no_lines
    # equipment_img = copy.copy(img)
    # get equipment text
    # assign it a hash color
    # find detecto boxes that intersect
    # use those boxes with that color
    # use equipment b
    # re_pattern = r'^[A-Z]{1,2}\d{5}-.*'
    ocr_equipment = get_text_in_ocr_results(re_pattern, ocr_results)  # where equipment = [box, name, score]

    # convert the line name into a color
    for equipment in ocr_equipment:
        print('looping through ocr equipment')
        equipment_id = equipment[1]

        # print(equipment_id)
        hashcolor = generate_color(equipment_id)
        equipment.append(hashcolor)

        # APPEND comment here?     #box is a detecto box
        dbox = convert_ocr_box_to_detecto_box(equipment[0])
        equipment_text_comment = get_comment(ocr_results, dbox, 20, include_inside=False)
        equipment.append(equipment_text_comment)



        #share

        print(equipment)
        # [[(2671, 106), (2863, 106), (2863, 148), (2671, 148)], 'PP-1130', 0.9728249629949269, (123, 187, 216), 'PP-1130~PS1 PIPELINE PUMPS~']
    # detecto equipment boxes, ocr equipment, extension range
    # now OCR equipment will have a extra comment appended

    share_equipment_comments(ocr_equipment)

    #make_detecto_equipment_boxes_black(processed_img,detecto_equipment_boxes)

    ocr_equipment = update_ocr_box(detecto_equipment_boxes, ocr_equipment, equipment_box_expand)

    #add_overlapping boxes to ocr equipment the idea is to merge certain boxes like if you have L shaped tank
    # but it might be broken because some equipment doens'
    #add_overlapping_boxes(ocr_equipment, detecto_equipment_boxes)

    # should we make a shrink coppy?
    for equipment in ocr_equipment:
        print('equipment debug', equipment)
        equipment[0] = [[int(x / shrink_factor), int(y / shrink_factor)] for x, y in equipment[0]]
    # we put the seed pixel locations in
    # note, x1, y1, x2, y2 = line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]
    seed_pixel_boxes = []
    for equipment in ocr_equipment:
        pixel_box = [equipment[0], equipment[3]]  # derived from easyocr [box, label, score, color]
        seed_pixel_boxes.append(pixel_box)

    # print(seed_pixels)
    print("doing flood fill evenly")
    # generate the colored line image
    equipment_img = multi_region_flood_fill(seed_pixel_boxes, processed_img, 1)

    # line_img = cv2.resize(line_img, None, fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_LINEAR)

    return (ocr_equipment, equipment_img)

def generate_painted_img(processed_img, ocr_results, colors, shrink_factor, expand_scale=1):
    # re_pattern = r'.*\"-[A-Z]{2,3}-[A-Z\d]{3,5}-.*'
    # re_pattern = r'.*\"-[A-Z]{1,5}-[A-Z\d]{3,5}-.*'
    #rocessed_service_img, lines, lines_with_service_in, shrink_factor,

    scale_results = copy.deepcopy(ocr_results)

    # also note that easy ocr gives 4 points as a box. So line[0] = p1, p2, p3, p4 where each point is a [x,y] pair
    # eg. line[0] = [[17, 5], [71, 5], [71, 17], [17, 17]]
    # have the boxes change size to reflect, eventually we might be able to skip this step
    for result in scale_results:
        result[0] = [[int(x / shrink_factor), int(y / shrink_factor)] for x, y in result[0]]
        # print("after "+str(line[0]))

    # we put the seed pixel locations in
    # note, x1, y1, x2, y2 = line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]
    seed_pixel_boxes = []
    for result, color in zip(scale_results, colors):
        pixel_box = [result[0], color]  # [box, color]
        seed_pixel_boxes.append(pixel_box)

    # print(seed_pixels)
    print("doing flood fill evenly")
    # generate the colored line image
    colored_img = multi_region_flood_fill(seed_pixel_boxes, processed_img, expand_scale)
    return colored_img

def return_serviced_lines(lns, lcolors, srvs, scolors, option):
    lines_with_service = []
    lines_with_service_color = []
    lines_with_service_txt = []
    services = []
    #note list index is being utilized
    for ln, lcolor in zip(lns, lcolors):
        for srv, scolor in zip(srvs, scolors):
            print(srv)
            stxt, slabel, sbox = srv
            if option == slabel:#service_in or service_out
                print(f'lcolor {lcolor}, scolor {scolor}')
                if lcolor == tuple(scolor):
                    #service = stxt
                    #ln[1] = stxt
                    lines_with_service.append(ln) # jsut for painting image
                    lines_with_service_color.append(lcolor)
                    lines_with_service_txt.append(stxt)
                    break

    return lines_with_service, lines_with_service_color, lines_with_service_txt

def return_hash_colors(ocr_results):
    # convert the line name into a color and fold it into the line object
    colors = []
    for result in ocr_results:
        result_id = result[1]
        print(result_id)
        hash_color = generate_color(result_id)
        colors.append(hash_color)
    return colors

def share_equipment_comments(ocr_equipment):
    seen = {}

    for equipment in ocr_equipment:
        label = equipment[1]
        comment = equipment[4]

        if label not in seen:
            seen[label] = comment
        else:
            seen[label] += '~' + comment

    for equipment in ocr_equipment:
        label = equipment[1]
        if label in seen:
            equipment[4] = seen[label]

def update_ocr_box(detecto_equipment_boxes, ocr_equipment, expand_amount):
    # data format of detecto box: x1d = int(box[0])    y1d = int(box[1])    x2d = int(box[2])    y2d = int(box[3])
    # data format of ocr box: x1o, y1o, x2o, y2o = ocr_equipment[0][0][0], ocr_equipment[0][0][1], ocr_equipment[0][2][0], ocr_equipment[0][2][1]
    # if a detecto box when expanded touches a ocr box, replace that ocr box with the detecto box
    def expand(dbox, amount):
        expanded_box = []
        expanded_box += [int(dbox[0] - amount)]  # x1
        expanded_box += [int(dbox[1] - amount)]  # y1
        expanded_box += [int(dbox[2] + amount)]  # x2
        expanded_box += [int(dbox[3] + amount)]  # y2
        return expanded_box

    def overlap(dbox, obox):
        x1d, y1d, x2d, y2d = dbox
        x1o, y1o, x2o, y2o = obox[0][0], obox[0][1], obox[2][0], obox[2][1]
        # Check for x-axis overlap
        x_overlap = (x1d <= x2o) and (x2d >= x1o)
        # Check for y-axis overlap
        y_overlap = (y1d <= y2o) and (y2d >= y1o)
        # If there is overlap on both axes, the boxes overlap
        if x_overlap and y_overlap:
            return True
        else:
            return False

    def shift_box(dbox, obox):
        print(f'shifting {dbox} to {obox}')
        x1d, y1d, x2d, y2d = dbox
        # Update each point in obox with the new coordinates
        obox[0] = [x1d, y1d]
        obox[1] = [x2d, y1d]
        obox[2] = [x2d, y2d]
        obox[3] = [x1d, y2d]

        return obox

    # Initialize a list to keep track of detecto boxes that have been matched
    matched_detecto_boxes = []
    matched_ocr_boxes = []
    for i, dbox in enumerate(detecto_equipment_boxes):
        expanded_dbox = expand(dbox, expand_amount)
        # matched = False

        for n, equipment in enumerate(ocr_equipment):
            if overlap(expanded_dbox, equipment[0]):
                equipment[0] = shift_box(dbox, equipment[0])
                matched_detecto_boxes.append(i)
                matched_ocr_boxes.append(n)
                # matched = True
                break


    print(f'all ocr equip boxes: {ocr_equipment}')
    #remove unmatched ocr boxes
    ocr_equipment = [ocr for i, ocr in enumerate(ocr_equipment) if i in matched_ocr_boxes]
    print(f'trimed ocr equip boxes: {ocr_equipment}')

    # Iterate over unmatched detecto boxes and append them as new OCR equipment
    extra_ocr_equipment = []
    for i, dbox in enumerate(detecto_equipment_boxes):
        if i not in matched_detecto_boxes:  # for unmatched equipment boxes
            for equipment in ocr_equipment:  # if one of the ocr_equipment (since boxes for some have been grown) overlaps
                if overlap(dbox, equipment[0]):
                    equip2 = equipment.copy()  # make a copy of the ocr equipment
                    equip2[0] = shift_box(dbox, equipment[0])  # change the box position
                    extra_ocr_equipment.append(equip2)  # append it to this list

    ocr_equipment += extra_ocr_equipment  # merege lists

    return ocr_equipment

def flood_fill_till_inst_reached(start, tag_no, data, img, shrink_factor):
    # do we need to make a copy of each image?

    start_time = time.time()
    # Create a new image to perform the flood fill on
    # img = copy.deepcopy(line_img)
    # Get the size of the image
    color = (0, 255, 0)
    height, width, channels = img.shape

    start = [int(value / shrink_factor) for value in start]
    # Make a corresponding list or queue and put the initial pixel in it
    active_pixel_list = []
    for pixel in [start]:
        index = []
        index.append(pixel)
        # Add the initial pixel to the active pixel list
        active_pixel_list.append(index)
        x, y = pixel
        print("x:", x, "y:", y)
        # Change the colors on the filled image to the seed colors
        img[y, x] = color

    # While there is still an element in any of the lists:
    # An empty list returns false
    while active_pixel_list:

        if time.time() - start_time > 2:
            print("Timeout reached.")
            return False

            # Enumerate gives us an index n
        for n, pixel_stack in enumerate(active_pixel_list):
            # print("chekcing stack "+str(n))
            start_size = len(pixel_stack)
            # i goes from the bottom to the top
            for i in range(start_size):
                pixel = active_pixel_list[n].pop(0)
                x, y = pixel

                # if curr_x, curr_y in a region break the loop
                for element in data:
                    if element['label'] == 'inst':
                        # shirnk box
                        # print('inst')
                        box = [int(value / shrink_factor) for value in element['box']]
                        if point_in_box(x, y, box):
                            # print(element[9])
                            element['sis'] = element['sis'] + f'{tag_no} '
                            print('found a sis inst pair!')
                            # assuming data is mutable
                            return True

                # Get neighbors
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for neighbor in neighbors:
                    nx, ny = neighbor
                    # Check if neighbor is within image bounds
                    if (0 <= nx < width) and (0 <= ny < height):
                        # Get the color of the neighboring pixel
                        neighbor_color = tuple(img[ny, nx])
                        # If the neighboring pixel is black
                        if neighbor_color == (0, 0, 0):
                            # Add pixel to pixel list
                            # print("append")
                            active_pixel_list[n].append([nx, ny])
                            # Set color on filled image
                            img[ny, nx] = color

        # Remove empty sublists
        active_pixel_list = [x for x in active_pixel_list if x]

def process_image(img, erosion_kernel, shrink_factor, lower_thresh):
    print('copying img')
    line_img = copy.copy(img)

    # Convert the RGB image to grayscale
    print('to grayscale')
    line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to get a binary image
    print('thresholding')
    _, line_img = cv2.threshold(line_img, lower_thresh, 255, cv2.THRESH_BINARY)

    # Convert the binary image back to RGB
    print('converting back to rgb')
    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)

    # Define erosion kernel
    # kernel_size = 12
    kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)

    # Perform erosion
    print('doing erosion')
    line_img = cv2.erode(line_img, kernel, iterations=1)

    # FOR FASTER PROCESSING, NOTE HALF THE ORIGINAL SIZE
    # shrink_factor = 6
    line_img = cv2.resize(line_img, None, fx=1 / shrink_factor, fy=1 / shrink_factor, interpolation=cv2.INTER_LINEAR)

    return line_img

def convert_ocr_box_to_detecto_box(obox):
    dbox = np.zeros(4)  # Initialize a NumPy array of size 4 with zeros
    dbox[0] = obox[0][0]
    dbox[1] = obox[0][1]
    dbox[2] = obox[2][0]
    dbox[3] = obox[2][1]
    return dbox

def get_box_center(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


    def append_data_slow(self, excel_type='xlwings'):
        """
        Append data to Excel file using either xlwings or openpyxl.

        Args:
            excel_type (str): Either 'xlwings' or 'openpyxl'
        """
        self.get_data_from_window()

        # Initialize workbook and worksheet
        if excel_type == 'xlwings':
            if not os.path.exists(self.workbook_path):
                self.wb = xw.Book()
                self.wb.save(self.workbook_path)
            wb = xw.Book(self.workbook_path)
            if 'Instrument Index' not in wb.sheet_names:
                wb.sheets.add(name='Instrument Index')
            ws = wb.sheets['Instrument Index']
        else:  # openpyxl
            if not os.path.exists(self.workbook_path):
                self.wb = openpyxl.Workbook()
            elif not self.wb:
                self.wb = openpyxl.load_workbook(self.workbook_path)
            if 'Instrument Index' not in self.wb.sheetnames:
                self.ws = self.wb.create_sheet('Instrument Index')
            else:
                self.ws = self.wb['Instrument Index']
            ws = self.ws

        print('we are begining to iterate though and update the inst data within append_data')
        for data in self.inst_data:
            print('data before: ', data)
            # Initialize default data dictionary
            data.update({
                'PID': self.pid,
                'SERVICE': '',
                'LINE/EQUIP': '',
                'INPUT COMMENT': self.comment if self.comment else '',
                'FILE': self.image_path,
            })
            # Process service information
            si = condense_hyphen_string(self.service_in)
            so = condense_hyphen_string(self.service_out)

            if self.service_in and self.service_out:
                data['SERVICE'] = f"{si} TO {so}" if self.service_in != self.service_out else f"{si} RECIRCULATION"
            elif self.service_in:
                data['SERVICE'] = f"{si} OUTLET"
            elif self.service_out:
                data['SERVICE'] = f"TO {so}"

            # Process line/equipment information
            if self.line:
                data['LINE/EQUIP'] = self.line
            elif self.equipment:
                words = self.equipment.split(' ')
                data['LINE/EQUIP'] = words[0]
                data['SERVICE'] = ' '.join(words[1:])

            # Get last row and handle header
            if excel_type == 'xlwings':
                last_row = ws.range('A1').expand('down').last_cell.row if ws.range('A1').value is not None else 0
            else:
                last_row = len(list(ws.rows))

            if (excel_type == 'xlwings' and ws.range('A1').value is None) or \
                    (excel_type == 'openpyxl' and ws['A1'].value is None):
                self.create_excel_header(ws, data)
                last_row += 1

            print('data after: ',data)

            self.populate_excel_row(ws, data, last_row + 1)

        if excel_type == 'openpyxl':
            self.wb.save(self.workbook_path)

        self.turn_boxes_blue()
