from IPython.display import display, HTML

def combine_test_file_labels_with_get_preds_results(
        learner_vocab,
        the_test_files, 
        predicted_probabilities, 
        predicted_class_indices):
    # get the highest prediction for each test set item
    highest_predicted_probs = [max(p) for p in predicted_probabilities]
    # get the predicted class/label/category of those highest predictions
    predicted_class_labels = [learner_vocab[pci] for pci in predicted_class_indices]
    # get the actual labels from their containing folders
    test_file_labels = [f.parent.stem for f in the_test_files]
    # zip it all up for ['predicted label', 'actual label', TensorBase(predicted probability)],
    predicted_actual_probability_list = [list(x) for x in zip(predicted_class_labels, test_file_labels, highest_predicted_probs)]
    
    return predicted_actual_probability_list

def create_prediction_result_div(label_and_result):
    return f"<div class='test-set-prediction-info'>{label_and_result[0]}/{label_and_result[1]}/{float(label_and_result[2]):.4f}</div>"

def create_img_tag(predicted, actual, img):
    cls = ''
    if predicted != actual:
        # add coloured border around image (see stylesheet variable)
        cls = 'test-set-prediction-incorrect'
    # only inline styling seems to work
    # use title for hover over reveal of file name
    return f"<img src={img} style='height:200px;width:200px;margin:auto;' title={img.name} class={cls}>"

def show_decoded_test_set_results(
        learner_vocab,
        test_files,
        predicted_probabilities,
        predicted_class_indices):

    labels_and_results = combine_test_file_labels_with_get_preds_results(
        learner_vocab, test_files, predicted_probabilities, predicted_class_indices)

    # inline styling easier than external css file dependency
    # and in some cases only inline rules are applied
    stylesheet = '<style> .test-set-grid-container { display: grid; grid-template-columns: auto auto auto; padding:8px;}'\
        ' .test-set-item { text-align: center; background-color: #fff;}'\
        ' .test-set-prediction-info { font-size: 1.75rem; font-weight: bold; padding-top: 8px; }'\
        ' .test-set-prediction-incorrect {border: 4px dashed #DC143C;}</style>'
    legend = '<div style="text-align:center;"><h2>Prediction/Actual/Probability</h2></div>'
    grid_container_start = '<div class="test-set-grid-container">'
    prediction_items_html = ''
    grid_container_end = '</div>'
    
    for i, lar in enumerate(labels_and_results):
        prediction_items_html += f"<div class='test-set-item'>{create_prediction_result_div(lar)}{create_img_tag(lar[0], lar[1], test_files.items[i])}</div>"
    
    test_set_html = f"{stylesheet}{legend}{grid_container_start}{prediction_items_html}{grid_container_end}"

    display(HTML(test_set_html))
