
import pyximport

from config import COCO_SGS_dir

pyximport.install(language_level=3)
from competitors.charts import plot_comparison_both, plot_comparison_coverage, plot_comparison_diversity



from competitors.kmeodids import load_resnet_images, compute_BOW_descriptors, read_BOW_images, kmedoids_summary, get_kmedoids_graphs, \
    competitors_dir

from sims.prs import edge_pruning, node_pruning, load_PRS
from sims.scene_graphs.image_processing import getImageName
from shutil import copyfile
import os

import pandas as pd
import json

from sims.graph_algorithms import compute_coverage_matrix
from sims.sgs_evaluation import evaluate_summary_graphs
from sims.sims_config import SImS_config
from sims.visualization import print_graphs

if __name__ == "__main__":
    class RUN_CONFIG:
        compute_BOW_descriptors = False # Map each COCO image to its BOW descriptors
        run_kmedoids = True             # Run KMedoids summary for different k values
        print_kmedoids_graphs = True   # Print scene graphs of selected kmedoids images (for each k)
        
        features = "resnet" # BOW or resnet -- depending on the feature extractor to be used
        
        include_in_eval = ["BOW", "resnet"]

        use_full_graphs = False         # True if you want to compute coverage on full graphs
                                        # False to apply node and edge pruning before computing coverage
        pairing_method = 'img_min'  # Method used to associate images to SGS graphs (see associate_img_to_sgs() in sgs.pyx)
                                     # img_min, img_max, img_avg, std

        compute_kmedoids_coverage_matrix = True # Compute graph coverage matrix for kmedoids
        evaluate_kmedoids = True  # Use coverage matrix to compute graph coverage and diversity of kmedoids
        write_comparison_charts = True # Write comparison charts between kmedoids and SImS (for white paper)

        #dataset = 'COCO_subset'     # Choose dataset
#         dataset = 'COCO_subset2'
        dataset = 'COCO_subset2'

        if dataset == 'COCO_subset':
            mink = 4                        # Min value of k to test kmedoids
            maxk = 20                       # Max value of k to test kmedoids
        elif dataset in ('COCO_subset2', 'COCO_subset3'):
            mink = 2
            maxk = 20

    # Paths:
    
    prefix = ""
    if RUN_CONFIG.features == "resnet":
        prefix = "resnet_"
    dsname_prefix = []
    output_path = os.path.join(competitors_dir, RUN_CONFIG.dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    kmedoids_out_clusters_path = os.path.join(output_path, f"{prefix}centroids.json")
    config = SImS_config(RUN_CONFIG.dataset)
    # --------------------------

    # Feature extraction for each image in COCO training set
    if RUN_CONFIG.compute_BOW_descriptors:
        compute_BOW_descriptors()
    # KMedoids summary for different k values
    if RUN_CONFIG.run_kmedoids:
        if RUN_CONFIG.features == "BOW":
            X = read_BOW_images(RUN_CONFIG.dataset)
        else:
            X = load_resnet_images(RUN_CONFIG.dataset)
        res = {}
        avg_time = 0
        print(f"Number of images: {len(X)}")
        for k in range(RUN_CONFIG.mink, RUN_CONFIG.maxk+1):
            medoids, duration = kmedoids_summary(X, k, RUN_CONFIG.features == "resnet") # with resnet, use cosine distance
            res[k] = (medoids, duration.seconds)
            avg_time += duration.seconds
            print(f"{k}: {medoids}")
            with open(os.path.join(output_path, "log.txt"),'a+') as f:
                f.write(f"{k}: {medoids}\n")
        print(str(avg_time/len(res)))
        with open(os.path.join(output_path, "avgTime.txt"), 'w') as f:
            f.write('Average time for kmedoids run on COCO subset (seconds):\n')
            f.write(str(avg_time/len(res)))
        with open(kmedoids_out_clusters_path,'w') as f:
            json.dump(res, f)
    # Print graphs associated to kmedoids
    if RUN_CONFIG.print_kmedoids_graphs:
        with open(kmedoids_out_clusters_path) as f:
            kmedoids_result = json.load(f)
        with open(config.scene_graphs_json_path, 'r') as f:
            coco_graphs = json.load(f)
        kmedoids_graphs = get_kmedoids_graphs(kmedoids_result, coco_graphs)

        for k, graphs in kmedoids_graphs.items():
            out_graphs_dir = os.path.join(output_path,'kmedoids_graphs',f'k{k}')
            if not os.path.exists(out_graphs_dir):
                os.makedirs(out_graphs_dir)
            print_graphs(graphs, out_graphs_dir)
            for i, g in enumerate(graphs):
                imgName = getImageName(g['graph']['name'], extension='jpg')
                copyfile(os.path.join(config.img_dir, imgName), os.path.join(out_graphs_dir, f"g{i}.jpg"))

    # Compute graph coverage for kmedoids (coverage matrix)
    if RUN_CONFIG.compute_kmedoids_coverage_matrix:
        with open(kmedoids_out_clusters_path) as f:
            kmedoids_result = json.load(f)
        with open(config.scene_graphs_json_path, 'r') as f:
            coco_graphs = json.load(f)
        kmedoids_graphs = get_kmedoids_graphs(kmedoids_result, coco_graphs)

        # Load pairwise relationship summary (PRS) if needed
        if RUN_CONFIG.use_full_graphs==False:
            prs = load_PRS(config, True)
        # Apply node and edge pruning before computing coverage matrix
        cmatrices_list = []
        omatrices_list = []
        for k, summary_graphs_i in kmedoids_graphs.items():
            if RUN_CONFIG.use_full_graphs == False:
                summary_graphs_i = edge_pruning(prs, summary_graphs_i)
                summary_graphs_i = node_pruning(summary_graphs_i)

            cmatrix, omatrix = compute_coverage_matrix(coco_graphs, [{'g':s} for s in summary_graphs_i])
            cmatrix.columns = list(range(int(k)))
            omatrix.columns = list(range(int(k)))
            cmatrix['k'] = k
            omatrix['k'] = k
            cmatrices_list.append(cmatrix)
            omatrices_list.append(omatrix)
        cmatrices = pd.concat(cmatrices_list, sort=True)
        omatrices = pd.concat(omatrices_list, sort=True)
        cmatrices.set_index('k', inplace=True)
        omatrices.set_index('k', inplace=True)
        cmatrices.index.name = 'k'
        omatrices.index.name = 'k'
        
        if RUN_CONFIG.use_full_graphs:
            output_file_c = os.path.join(output_path, f"{prefix}coverage_mat_full.csv")
            output_file_o = os.path.join(output_path, f"{prefix}overlap_mat_full.csv")
        else:
            output_file_c = os.path.join(output_path, f"{prefix}coverage_mat_pruned.csv")
            output_file_o = os.path.join(output_path, f"{prefix}overlap_mat_pruned.csv")
        cmatrices.to_csv(output_file_c, sep=",")
        omatrices.to_csv(output_file_o, sep=",")

    # Compute coverage and diversity for kmedoids
    if RUN_CONFIG.evaluate_kmedoids:
        with open(kmedoids_out_clusters_path) as f:
            kmedoids_result = json.load(f)
        with open(config.scene_graphs_json_path, 'r') as f:
            coco_graphs = json.load(f)
        kmedoids_graphs = get_kmedoids_graphs(kmedoids_result, coco_graphs)

        if RUN_CONFIG.use_full_graphs==False:
            suffix = "_pruned"
            suffix2=f"_{RUN_CONFIG.pairing_method}"
        else:
            suffix = "_full"
            suffix2 = ""
        
        cmatrices = pd.read_csv(os.path.join(output_path, f"{prefix}coverage_mat{suffix}.csv"), index_col='k')
        omatrices = pd.read_csv(os.path.join(output_path, f"{prefix}overlap_mat{suffix}.csv"), index_col='k')

        # Load pairwise relationship summary (PRS) if needed
        if RUN_CONFIG.use_full_graphs==False:
            prs = load_PRS(config, True)
        # Prune kmedoids graphs before computing coverage and diversity
        results = []
        for k, summary_graphs_i in kmedoids_graphs.items():
            if RUN_CONFIG.use_full_graphs == False:
                summary_graphs_i = edge_pruning(prs, summary_graphs_i)
                summary_graphs_i = node_pruning(summary_graphs_i)

            res = evaluate_summary_graphs([{'g':s} for s in summary_graphs_i], cmatrices.loc[int(k)].iloc[:,:int(k)], omatrices.loc[int(k)].iloc[:,:int(k)])
            results.append(res)

        kmed_df = pd.DataFrame(results, columns=["N. graphs",
                                                "Avg. nodes", "Std. nodes",
                                                "Coverage",
                                                "Coverage-overlap",
                                                "Diversity",
                                                "Diversity-ne"])
        kmed_df.to_csv(os.path.join(output_path, f"{prefix}evaluation{suffix}.csv"))

    # Write comparison charts for white paper
    if RUN_CONFIG.write_comparison_charts:
        if RUN_CONFIG.use_full_graphs==False:
            suffix = "_pruned"
            suffix2=f"_{RUN_CONFIG.pairing_method}"
        else:
            suffix = "_full"
            suffix2 = ""

        # Read SImS results
        
        dfs_to_eval = {}
        
        sims_eval_path = os.path.join(config.SGS_dir, f'evaluation{suffix2}.csv')
        if not os.path.exists(sims_eval_path):
            print(sims_eval_path, "not found. You have to evaluate SImS first. Run main_SGS.py with evaluate_SGS_experiments=True")
            exit()
        else:
            sims_df = pd.read_csv(sims_eval_path, index_col=0)
            dfs_to_eval["SImS"] = sims_df

        # Read kmedoids results
    
        if "BOW" in RUN_CONFIG.include_in_eval:
            kmed_eval_path = os.path.join(output_path, f"evaluation{suffix}.csv")
            if not os.path.exists(kmed_eval_path):
                print("You have to evaluate Kmedoids with BOW first. Run main_competitors.py with evaluate_kmedoids=True")
                exit()
            else:
                kmed_df = pd.read_csv(kmed_eval_path, index_col=0)
                dfs_to_eval["SIFT"] = kmed_df

        if "resnet" in RUN_CONFIG.include_in_eval:
            kmed_eval_path = os.path.join(output_path, f"resnet_evaluation{suffix}.csv")
            if not os.path.exists(kmed_eval_path):
                print("You have to evaluate Kmedoids with ResNet first. Run main_competitors.py with evaluate_kmedoids=True and features=resnet")
                exit()
            else:
                kmed_df = pd.read_csv(kmed_eval_path, index_col=0)
                dfs_to_eval["ResNet50"] = kmed_df


        sims_agg_df = None
        plot_comparison_both(RUN_CONFIG.mink, RUN_CONFIG.maxk, dfs_to_eval, suffix2, output_path)
        plot_comparison_coverage(RUN_CONFIG.mink, RUN_CONFIG.maxk, dfs_to_eval, suffix2, output_path)
        plot_comparison_diversity(RUN_CONFIG.mink, RUN_CONFIG.maxk, dfs_to_eval, suffix2, output_path)
