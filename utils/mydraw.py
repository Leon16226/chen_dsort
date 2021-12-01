from deepsort.utils import compute_color_for_id, plot_one_box


def show_boxes_draw(real_box, s_im0s, labels, width, height):
    for box in real_box:
        bboxes = box[0:4]
        bboxes[[0, 2]] = (bboxes[[0, 2]] * width).round()
        bboxes[[1, 3]] = (bboxes[[1, 3]] * height).round()
        cls = box[4]
        conf = box[5]
        cls = int(cls)

        label = f'{labels[cls]}{conf:.2f}'
        color = compute_color_for_id(cls)
        plot_one_box(bboxes, s_im0s, label=label, color=color, line_thickness=2)
