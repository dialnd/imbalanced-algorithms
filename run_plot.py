from algorithms.utils import _read_split,_class_split
from visual.plot import plot_data,plot_syn,plot_X
from algorithms.smote import SMOTE
import warnings

trX, teX, trY, teY = _read_split(
	"../datasets/nd-data/segment.csv",
	read=1,oneHot=0)

C0,C1=_class_split(trX,trY)
warnings.filterwarnings("ignore", category=DeprecationWarning)
syn=SMOTE(C1,100,5)

#to analyze training data
plot_data(trX,trY)

#to compare synthetic samples
plot_syn(C1,syn)

#to analyze the class
plot_X(C1)
