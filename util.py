import os
import pickle
class Project:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._init_all_path()


    def _init_all_path(self):
        self._data_dir = os.path.join(self._root_dir, 'data')
        self._aux_data_dir = os.path.join(self._data_dir, 'auxx')
        self._preprocessed_data_dir = os.path.join(self._data_dir,'preprocessed')
        self._feature_dir = os.path.join(self._data_dir, 'feature')
        self._trained_model_dir = os.path.join(self._data_dir, 'model')
        self._tmp_dir = os.path.join(self._data_dir, 'tmp')

    @property
    def root_dir(self):
        return self._root_dir + os.path.sep

    @property
    def data_dir(self):
        return self._data_dir + os.path.sep

    @property
    def aux_dir(self):
        return self._aux_data_dir + os.path.sep

    @property
    def preprocessed_data_dir(self):
        return self._preprocessed_data_dir + os.path.sep

    @property
    def feature_dir(self):
        return self._feature_dir + os.path.sep

    @property
    def trained_model_dir(self):
        return self._trained_model_dir + os.path.sep

    @property
    def tmp_dir(self):
        return self._tmp_dir + os.path.sep

    @staticmethod
    def init(root_dir, create_dir=True):
        project = Project(root_dir)
        if create_dir:
            path_to_creat = [
                project.data_dir,
                project.aux_dir,
                project.preprocessed_data_dir,
                project.feature_dir,
                project.trained_model_dir,
                project.tmp_dir
            ]
            for path in path_to_creat:
                if os.path.exists(path):
                   continue
                else:
                    os.makedirs(path)
        return project

    # def save(self,nfile,object):
    #     with open(nfile, 'wb') as file:
    #         pickle.dump(object, file)
    def save(self, nfile, object):
        with open(nfile, 'wb') as file:
            pickle.dump(object, file=file)

    def load(self, nfile):
        with open(nfile, 'rb') as file:
            return pickle.load(file)

    def save_feature_names(self, fea_names, feature_name):
        self._save_feature_col_name(fea_names, self.feature_dir + 'X_train_{}.names'.format(feature_name))

    def save_feature_col_list(self,fea_data,type,feature_name):
        # 保存各列对应的数据
        self._save_feature_data(fea_data,self.features_dir + 'X_{}_{}.pickle'.format(type,feature_name))

    def _save_feature_data(self, data, nfile):
        with open(nfile, 'wb') as file:
            pickle.dump(data, file)
    def _save_feature_col_name(self, col_names, nfile):
        with open(nfile, 'wb') as file:
            file.write('\n'.join(col_names))

    def _load_feature_data(self, nfile):
        with open(nfile, 'r') as file:
            return [line.rstrip('\n') for line in file.readlines()]

    def _load_feature_data(self, nfile):
        with open(nfile, 'rb') as file:
            return pickle.load(file)

    def save_features(self, train_fea, test_fea, fea_names, feature_name):
        self.save_feature_names(fea_names, feature_name)
        self.save_feature_col_list(train_fea, 'train', feature_name)
        self.save_feature_col_list(train_fea, 'test', feature_name)
        pass



if __name__ == '__main__':
    path = os.getcwd()
    Project.init(path, create_dir=True)
    print(path)