import time 

def stratified_sampling(dataframe, target_variable: str, target_classes: tuple, sample_size: float):
  '''this function performs a stratified random sampling of a given data set (for a binary class problem) respect to the unbalanced 
      class of the target variable keeping the same proportions of the minority class in the original dataframe inside the new-sampled one      
      : dataframe - a pandas dataframe
      : target_variable - the target variable in which is present the class where to stratify
      : target_classes - is a tuple, give the classes of the target variable as a string
      : sample_size - dimension of the sampled dataframe
      :return the final stratified sampled dataframe'''
  start = time.time()
  df_y = dataframe.loc[dataframe[target_variable] == target_classes[0]]
  df_n = dataframe.loc[dataframe[target_variable] == target_classes[1]]
  p_y = len(df_y)/(len(dataframe))
  p_n = 1 - p_y
  y_s = df_y.sample(frac = 1).iloc[:int(p_y*len(dataframe)*sample_size)]
  n_s = df_n.sample(frac = 1).iloc[:int(p_n*len(dataframe)*sample_size)]
  fdf = pd.concat([y_s, n_s], axis = 0, ignore_index = True).sample(frac =1)
  print(f'Time to process: {round(time.time()-start,3)}')
  return fdf
