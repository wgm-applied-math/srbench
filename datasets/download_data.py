import os
import pmlb
from pmlb import fetch_data, regression_dataset_names 
import yaml

print(pmlb.__version__)

ddir = '.'

print("Saving datasets to:", ddir)

dataset_categories = {
    'blackbox': ['1028_SWD', '1089_USCrime', '1193_BNG_lowbwt', '1199_BNG_echoMonths', 
                 '192_vineyard', '210_cloud', '522_pm10', '557_analcatdata_apnea1',
                 '579_fri_c0_250_5', '606_fri_c2_1000_10', '650_fri_c0_500_50',
                 '678_visualizing_environmental'],
    'firstprinciples': ['first_principles_absorption', 'first_principles_bode', 
                        'first_principles_hubble', 'first_principles_ideal_gas',
                        'first_principles_kepler', 'first_principles_leavitt',
                        'first_principles_newton', 'first_principles_planck',
                        'first_principles_rydberg', 'first_principles_schechter',
                        'first_principles_supernovae_zr',
                        'first_principles_tully_fisher']
}

for category_name, dataset_list in dataset_categories.items():
    category_path = f"{ddir}/{category_name}"
    os.makedirs(category_path, exist_ok=True)
    
    for dataset_name in dataset_list:
        try:
            print(f"\nDownloading {dataset_name} to {category_name} category...")
            dataset = fetch_data(dataset_name, local_cache_dir=category_path)

            print(f"Successfully downloaded {dataset_name}")
        except Exception as e:
            print(f"Error downloading {dataset_name}: {str(e)}")
        
        # Write metadata.yml file
        metadata = {
            "name": dataset_name,
            "task": "regression",
            "n_instances": dataset.shape[0],
            "n_features": dataset.shape[1]
        }

        dataset_folder = os.path.join(category_path, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        metadata_path = os.path.join(dataset_folder, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)