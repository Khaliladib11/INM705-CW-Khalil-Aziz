Traceback (most recent call last):
  File "Model_Training.py", line 57, in <module>
    train_interface = CityScapesInterface(**train_interface_params) # train data
  File "/users/adcm114/INM705/CW/CityScapes.py", line 48, in __init__
    self.cities = os.listdir(os.path.join(self.data_root, 'leftImg8bit', self.phase))
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/course/psarin/inm705/leftImg8bit/train'
