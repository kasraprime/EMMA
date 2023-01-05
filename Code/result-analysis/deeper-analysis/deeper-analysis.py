import pandas as pd
import numpy as np


metrics_objects = pd.read_csv('wandb_export_2022-12-19T14_42_16.465-05_00.csv')
diffs = {
'diff_lard': np.around(np.array(metrics_objects['EMMA_lard']) - np.array(metrics_objects['SupCon_lard']), 2),
'diff_lrd': np.around(np.array(metrics_objects['EMMA_lrd']) - np.array(metrics_objects['SupCon_lrd']), 2),
'diff_ard': np.around(np.array(metrics_objects['EMMA_ard']) - np.array(metrics_objects['SupCon_ard']), 2),
'diff_ar': np.around(np.array(metrics_objects['EMMA_ar']) - np.array(metrics_objects['SupCon_ar']), 2),
'diff_ad': np.around(np.array(metrics_objects['EMMA_ad']) - np.array(metrics_objects['SupCon_ad']), 2),
'diff_lr': np.around(np.array(metrics_objects['EMMA_lr']) - np.array(metrics_objects['SupCon_lr']), 2),
'diff_ld': np.around(np.array(metrics_objects['EMMA_ld']) - np.array(metrics_objects['SupCon_ld']), 2),
'diff_lar': np.around(np.array(metrics_objects['EMMA_lar']) - np.array(metrics_objects['SupCon_lar']), 2),
'diff_lad': np.around(np.array(metrics_objects['EMMA_lad']) - np.array(metrics_objects['SupCon_lad']), 2),
}


# sorted_diff_lard = sorted(list(zip(metrics_objects['objects'], diff_lard)), key=lambda x:x[1])
# sorted_diff_lrd = sorted(list(zip(metrics_objects['objects'], diff_lrd)), key=lambda x:x[1])
# sorted_diff_ard = sorted(list(zip(metrics_objects['objects'], diff_ard)), key=lambda x:x[1])

col = ['Objects']+[metr for metr in list(diffs.keys())]
diff_df = pd.DataFrame(columns=col)

for idx in range(len(list(metrics_objects['objects']))):
    diff_df.at[idx, 'Objects'] = metrics_objects['objects'][idx]
    for metr in list(diffs.keys()):
        diff_df.at[idx, metr] = diffs[metr][idx]
    
    # diff_df.at[idx, 'diff_lard'] = diff_lard[idx]
    # diff_df.at[idx, 'diff_lrd'] = diff_lrd[idx]
    # diff_df.at[idx, 'diff_ard'] = diff_ard[idx]


diff_df.to_csv(path_or_buf='diff_score_all.csv', index=False)

# for x in sorted_diff_lard:
#     print(f"{x[0]}: {x[1]} ")