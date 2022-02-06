import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import time
start_time = time.time()

number_of_breeds = 133
model_from_transfer = models.densenet161(pretrained=True)
number_inputs = model_from_transfer.classifier.in_features
last_layer = nn.Linear(number_inputs, number_of_breeds)
model_from_transfer.classifier = last_layer
model_from_transfer.load_state_dict(torch.load('learned_model_from_transfer.pt'))

classes = [ '001.Pinczer małpi', '002.Chart afgański', '003.Airedale terrier', '004.Akita', '005.Alaskan malamute', '006.Amerykański pies eskimoski', 
            '007.Foxhound amerykański', '008.Amerykański Staffordshire terrier', '009.Amerykański spaniel dowodny', '010.Anatolian', '011.Australijski cattle dog', 
            '012.Owczarek australijski', '013.Terier australijski', '014.Basenji', '015.Basset', '016.Beagle', '017.Bearded collie', 
            '018.Owczarek francuski beauceron', '019.Bedlington terrier', '020.Malinois', '021.Owczarek belgijski', '022.Tervueren', '023.Berneński pies pasterski', 
            '024.Bichon frise', '025.Black and tan coonhound', '026.Czarny terier rosyjski', '027.Bloodhound', '028.Bluetick coonhound', '029.Border collie', 
            '030.Border terrier', '031.Borzoj', '032.Boston terrier', '033.Bouvier des Flandres', '034.Bokser', '035.Boykin spaniel', '036.Owczarek francuski briard', 
            '037.Épagneul breton', '038.Gryfonik brukselski', '039.Bulterier', '040.Buldog', '041.Bulmastif', '042.Cairn terrier', '043.Canaan Dog', '044.Cane corso',
            '045.Welsh Corgi Cardigan', '046.Cavalier king charles spaniel', '047.Chesapeake Bay retriever', '048.Chihuahua', '049.Grzywacz chiński', '050.Shar pei', 
            '051.Chow chow','052.Clumber Spaniel', '053.Cocker spaniel angielski', '054.Owczarek szkocki długowłosy', '055.Curly coated retriever', '056.Jamnik', '057.Dalmatyńczyk', '058.Dandie Dinmont terrier', 
            '059.Doberman', '060.Dog z Bordeaux', '061.Cocker spaniel angielski', '062.Seter angielski', '063.Springer spaniel angielski', '064.English toy spaniel',
            '065.Entlebucher', '066.Field Spaniel', '067.Szpic fiński', '068.Flat coated retriever', '069.Buldog francuski', '070.Pinczer średni', 
            '071.Owczarek niemiecki', '072.Wyżeł niemiecki krótkowłosy ', '073.Wyżeł niemiecki szorstkowłosy', '074.Sznaucer olbrzym', '075.Irish glen of imaal terrier', 
            '076.Golden retriever', '077.Seter szkocki', '078.Dog niemiecki', '079.Pirenejski pies górski', '080.Duży szwajcarski pies pasterski', '081.Greyhound', '082.Hawańczyk', 
            '083.Podenco z Ibizy', '084.Islandzki szpic pasterski', '085.Seter irlandzki czerwono-biały ', '086.Seter irlandzki', '087.Terier irlandzki', '088.Irlandzki spaniel dowodny', 
            '089.Wilczarz irlandzki', '090.Charcik włoski', '091.Chin japoński', '092.Szpic wilczy', '093.Kerry blue terrier', '094.Komondor', '095.Kuvasz', 
            '096.Labrador retriever', '097.Lakeland terrier', '098.Leonberger', '099.Lhasa apso', '100.Lwi piesek', '101.Maltańczyk', '102.Manchester terrier', 
            '103.Mastif', '104.Sznaucer miniaturowy', '105.Mastif neapolitański', '106.Nowofundlandczyk', '107.Norfolk terrier', '108.Buhund norweski', '109.Elkhund szary', 
            '110.Norsk Lundehund', '111.Norwich Terrier', '112.Retriever z Nowej Szkocji', '113.Owczarek staroangielski', '114.Otterhound', '115.Papillon', 
            '116.Parson Russell terrier', '117.Pekińczyk', '118.Welsh Corgi Pembroke', '119.Petit Basset Griffon Vendéen', '120.Pies faraona', '121.Plott hound', '122.Pies Pointer', 
            '123.Szpic miniaturowy', '124.Pudel duży', '125.Portugalski pies dowodny', '126.Bernardyn', '127.Australijski silky terier', '128.Foksterier krótkowłosy', '129.Mastif tybetański', 
            '130.Springer spaniel walijski', '131.Gryfon Korthalsa', '132.Nagi pies meksykański', '133.Yorkshire terrier']

def predict_breed_transfer(img_path):
    model_from_transfer.eval()
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    img = Image.open(img_path).convert('RGB') # przypisanie piksela wartosci kanalow RGB (trzy wartosci z zakresu 0-255)
    img = transform(img).float() # uejdnolicenie obrazow
    img = img.unsqueeze(0) # dodatkowy wymiar do tensora, parametr ( 0 ) inforuje ze dodakowy wymiar jest na poczatku 
    

    predictions = model_from_transfer(img)
    top3 = predictions.topk(3)

    def get_index(i):
        return classes[top3.indices[0][i].item()]
    
    return get_index(0), get_index(1), get_index(2)


def classify(path):

    print("algorithm working\n")

    number_of_breeds = 133
    model_from_transfer = models.densenet161(pretrained=True)
    number_of_inputs = model_from_transfer.classifier.in_features
    last_layer = nn.Linear(number_of_inputs, number_of_breeds)
    model_from_transfer.classifier = last_layer
    model_from_transfer.load_state_dict(torch.load('learned_model_from_transfer.pt'))


    dogPath = path
    print("Your dog looks like: ")
    #print(predict_breed_transfer(path)[:])
    return predict_breed_transfer(path)[0][4:], predict_breed_transfer(path)[1][4:], predict_breed_transfer(path)[2][4:]
