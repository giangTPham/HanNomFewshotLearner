from models import TripletModel


model = TripletModel('resnet50', 256, False, True)

print(vars(model))