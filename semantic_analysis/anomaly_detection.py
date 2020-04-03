



def inspect_anomalies(image_graph, kb, pq_stat, anomaly_stat, no_istogram):

    nodes = {}
    for node in image_graph['nodes']:
        nodes[node['id']] = node['label']
    # Run for each link in the graph
    for link in image_graph['links']:
        sub = nodes[link['s']]
        ref = nodes[link['r']]
        pos = link['pos']
        pair = f"{sub},{ref}"
        selected_opt = None
        # Check for the likelihood in the KB
        l = None
        if pair in kb:
            hist = kb[f"{sub},{ref}"]
            if pos in hist:
                l = hist[pos]
            else:
                l = 0

        # Look for the different options
        if link['s'] in pq_stat['fp'] and link['r'] in pq_stat['fp']:
            selected_opt = 'both_fp'
        elif (link['s'] in pq_stat['tp'] and link['r'] in pq_stat['fp']) \
                or (link['r'] in pq_stat['tp'] and link['s'] in pq_stat['fp']):
            selected_opt = 'tp_fp'
        elif link['s'] in pq_stat['fp'] or link['r'] in pq_stat['fp']:
            selected_opt = 'fp_ignored'
        else:
            selected_opt = 'ignored'

        if l is not None:
            anomaly_stat[selected_opt]['l'].append(l)
            anomaly_stat[selected_opt]['sup'].append(hist['sup'])
            anomaly_stat[selected_opt]['entropy'].append(hist['entropy'])
        else:
            no_istogram[selected_opt]+=1


        #item = {(sub, ref): value for key, value in kb_pairwise.items() if sub in key and ref in key}
        # if len(item) != 1 or not item:
        #     continue
        # try:
        #     likelihood = item[(sub, ref)][pos]
        # except Exception:
        #     likelihood = None
        # support = item[(sub, ref)]['sup']
        # entropy = item[(sub, ref)]['entropy']
        # pair = str(link['s']) + "," + str(link['r'])
        # if image_id not in likelihoods.keys():
        #     likelihoods[image_id] = {'fp': fp_img["fp"],
        #                              'pairs': {pair: {'l': likelihood, 's': support, 'e': entropy}}}
        # else:
        #     likelihoods[image_id]['pairs'].update({pair: {'l': likelihood, 's': support, 'e': entropy}})