import argparse
import deep_impact.ml.dawn.model.model as model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        help='GCS or local paths',
        required=False,
        default=r"C:\github\deep_impact\bq\#local\dawn_modeldata_m.csv"
        #default=r"C:\github\deep_impact\bq\#local\dawn_modeldata_m_micro_test.csv"
    )
    parser.add_argument(
        '--eval-files',
        help='GCS or local paths',
        required=False,
        default=r"C:\github\deep_impact\bq\#local\dawn_modeldata_m.csv"
        #default=r"C:\github\deep_impact\bq\#local\dawn_modeldata_m_micro_test.csv"
    )
    parser.add_argument(
        '--output',
        help='GCS or local paths,学習したモデルの出力先',
        required=False,
        default=r"C:\github\deep_impact\bq\#local\models\dawn"
    )
    parser.add_argument(
        '--job-dir',
        help='GCS or local paths,学習したモデルの出力先,outputとダブっているのでゆくゆく統合したいが、local trainモードとml engineモードでのパラメータの違いが・・・',
        required=False,
        default = r"C:\github\deep_impact\bq\#local\models\dawn"
    )
    args = parser.parse_args()
    #input_processor_test(args)
    #model.input_processor_test(args)
    model.train(args)
