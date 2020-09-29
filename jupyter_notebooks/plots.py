import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

def avgStrain(strainTraj_df):
    out = []
    for t in range(90):
        tmp = strainTraj_df.xs('exz', level='values')['(0,{})'.format(t)].describe()
        out.append([t, tmp['mean'], tmp['std']])
    return pd.DataFrame(out,columns=['Frame Number', 'Mean affine shear strain <exz>', 'Standard deviation'])

def d2Min_joint_strain(strainTraj_df, tPairList,figPath, slice_keys={'x':'D2_min', 'y':'exz'}):
    """
    Still has to be updated to get axes labels, fileName, and filtering selection passed by dictionary
    """
    out = []
    for n in range(len(tPairList)):
        # for each pair or tCur and tRef, slice strainTraj to get D2Min and shearStrain
        tRef, tCur = tPairList[n]
        #d2_tmp = strainTraj_df.xs('D2_min',level='values')['({},{})'.format(tRef,tCur)]
        x_tmp = strainTraj_df.xs(slice_keys['x'],level='values')['({},{})'.format(tRef,tCur)]
        #exz_tmp = strainTraj_df.xs('exz',level='values')['({},{})'.format(tRef,tCur)]
        y_tmp = strainTraj_df.xs(slice_keys['y'],level='values')['({},{})'.format(tRef,tCur)]

        # Do some simple filtering to remove outliers
        #d2_idx = d2_tmp[d2_tmp<1.0].index
        x_idx = x_tmp[x_tmp<1.0].index
        #exz_idx = exz_tmp[(exz_tmp > -0.02) & (exz_tmp < 0.03)].index
        y_idx = y_tmp[(y_tmp > -0.02) & (y_tmp < 0.03)].index
        idx = x_idx.intersection(y_idx)

        # make the plot and save the figure
        sns.jointplot(y=y_tmp.loc[idx], x=x_tmp.loc[idx], kind='hex').set_axis_labels("Non-affinity parameter (D^2_min)",
                                                                                      "Local affine shear strain (e_xz)")
        fPath_save =figPath + '/{}'.format('jointPlot_exz_D2_min_t{:02}_t{:02}.png'.format(tRef, tCur))
        plt.savefig(fPath_save, dpi=300, bbox_inches='tight', metaData={'strain tRef': tRef,
                                                                        'strain tCur': t,
                                                                        'clipping exz': '>-0.02 & <0.03',
                                                                        'clipping D2 min': '<1'})
        plt.close()
        out.append(fPath_save)
     return out

def highD2Min_xyz(strainTraj_df, pos_df, tPairList, cutOff, outPathStem=None,fName_stem=None):
    if outPathStem == None:
        d2Min_xyzPath = '/Users/zsolt/Colloid/DATA/tfrGel10212018x/tfrGel10212018A_shearRun10292018f/plots/large_D2Min_xyz'
    if fName_stem == None:
        fName_stem = 'd2Min_refStrain_t{:02}.xyz'

    # using the strainTraj_df select for high D2_min particles
    # then change table format to match (frame,particle) multindex on pos_df
    tmp = (strainTraj_df.xs('D2_min', level='values') > cutOff).transpose().reset_index()
    tmp.index = range(1, 90)
    tmp = pd.DataFrame(tmp.drop(columns='index').stack())
    tmp.set_index(tmp.index.rename(['frame', 'particle']), inplace=True)
    tmp = tmp.rename(columns={0: 'Large D2_Min'})
    pos_df = pos_df.join(tmp)

    for t in range(1, 90):
        d2Min_fName = fName_stem.format(t)
        with open(d2Min_xyzPath + '/{}'.format(d2Min_fName), 'a') as f:
            pos_tmp = pos_df.xs(t, level='frame').dropna()[pos_df.xs(t, level='frame').dropna()['Large D2_Min']][
                ['x (um, imageStack)', 'y (um, imageStack)', 'z (um, imageStack)']]
            f.write('{}\n\n'.format(len(pos_tmp)))
            pos_tmp.to_csv(f, sep=' ', header=False, index=False)
    return outPathStem

def script_localStrain_heatMap(strain_df, pos_df,gelFitPlane_dict):
    """
    Code to make a heat map of local strain with the positions referenced to pos_df, even if the strain reference
    configuration changes
    """

    # get index of all particle within 15um of gel by selecting on position df as reference time
    distFromPlane(pos_df, 'height above gel (um)', gelFitPlane_dict)
    pos_t0 = pos_df.xs(0,level='frame').dropna()
    idx = pos_t0[pos_t0['height above gel (um)'] < 15].dropna().index

    # get index of all particle that have sufficient nnb to make strain valid and intersect with spatial selection above
    tmp = strain_df[strain_df.loc[(slice(None),'nnb count'), : ] >= 9].dropna().unstack()
    idx = idx.intersection(tmp.index)

    # carry out the selection above and select 'eyx' strain component
    buf = strain_df.unstack().reindex(idx)
    sedStrain_bottom = buf.loc[:, buf.columns.get_level_values(1) == 'eyz'].droplevel(1, axis=1)

    # bin in x and y
    sedStrain_bottom['xbin'] = pd.cut(pos0['x (um, imageStack)'], 40)
    sedStrain_bottom['ybin'] = pd.cut(pos0['y (um, imageStack)'], 40)
    sedStrain_bottom['zbin'] = pd.cut(pos0['height above gel (um)'], 3)

    tmp = sedStrain_bottom.groupby(['xbin', 'ybin']).mean().apply(lambda x: x * 100)
    for t in sedStrain_bottom.columns[0:-3]:
        sns.heatmap(tmp[t].unstack(), cmap='PiYG', vmin=-0.8, vmax=0.8)
        t_fName = 't{}'.format(t.replace('(', '').replace(')', '').replace(',', '_'))
        plt.title('Local shear strain\n averaged over ~5um square boxes')
        plt.savefig(figPath + '/{}/{}'.format('sedStrain/localStrain', 'localStrain_dt3_{}.png'.format(t_fName)),
                    dpi=300, bbox_inches='tight')
        plt.close()
    return True




