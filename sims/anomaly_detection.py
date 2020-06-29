import pyximport
pyximport.install(language_level=3)
from sims.prs import get_likelihood


def inspect_anomalies2(image_graph, kb, pq_stat, anomaly_stat, thr):

    nodes = {}
    objects = {}
    for node in image_graph['nodes']:
        nodes[node['id']] = node['label']
        objects[node['id']] = {'n_links':0, 'n_anom':0, 'l':0}
    # Run for each link in the graph
    for link in image_graph['links']:
        l,_ = get_likelihood(nodes, link, kb)

        if l is not None:
            if l<thr:
                objects[link['s']]['n_anom']+= 1
                objects[link['r']]['n_anom']+= 1
            objects[link['s']]['n_links'] += 1
            objects[link['r']]['n_links'] += 1
            objects[link['s']]['l'] += l
            objects[link['r']]['l'] += l

    for obj_id, stat in objects.items():
        tpfp = None
        if obj_id in pq_stat['fp']:
            tpfp = 'fp'
        elif obj_id in pq_stat['tp']:
            tpfp = 'tp'
        if tpfp is not None:
            anomaly_stat[tpfp]['n_anom'].append(stat['n_anom'])
            anomaly_stat[tpfp]['n_links'].append(stat['n_links'])
            if stat['n_links']>0:
                anomaly_stat[tpfp]['perc_anom'].append(stat['n_anom']/stat['n_links'])
                anomaly_stat[tpfp]['avg_l'].append(stat['l']/stat['n_links'])
            else:
                anomaly_stat[tpfp]['perc_anom'].append(0)
                anomaly_stat[tpfp]['avg_l'].append(0)


def inspect_anomalies(image_graph, kb, pq_stat, anomaly_stat, no_istogram):

    # Node map
    nodes = {node['id']: node['label'] for node in image_graph['nodes']}

    # Run for each link in the graph
    for link in image_graph['links']:
        l,hist = get_likelihood(nodes, link, kb)

        # Look for the different options
        if link['s'] in pq_stat['fp'] and link['r'] in pq_stat['fp']:
            selected_opt = 'both_fp'
        elif link['s'] in pq_stat['tp'] and link['r'] in pq_stat['tp']:
            selected_opt = 'both_tp'
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