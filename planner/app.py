from typing import Literal

from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("index.html")


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse("favicon.ico")


class Params(BaseModel):
    object_type: Literal["Школа", "Больница"]


@app.post("/calculate")
async def calculate(params: Params):
    heat_layer = [
        [55.72624935999874, 37.63248699716266, 0.11234721407624633],
        [55.7261756259539, 37.64015824936847, 0.19164809384164222],
        [55.7261014022252, 37.64782946787556, 0.29788152492668624],
        [55.72602669048027, 37.65550065062936, 0.11093489736070382],
        [55.72587580402166, 37.67084291174203, 0.2864140762463343],
        [55.73088704151838, 37.61726825475894, 0.12944046920821115],
        [55.730814271041275, 37.624940408852794, 0.08558826979472141],
        [55.73074101134778, 37.632612528770665, 0.10323988269794722],
        [55.73066726353914, 37.64028461431471, 0.1333067448680352],
        [55.73059302815935, 37.647956666208835, 0.36370674486803517],
        [55.73051830523145, 37.65562868235247, 0.15521407624633432],
        [55.730443093106615, 37.663300663407114, 0.0621466275659824],
        [55.73036739394563, 37.670972612952944, 0.30312727272727275],
        [55.73029120563449, 37.67864452320777, 0.11084105571847508],
        [55.7354510075472, 37.609719130878794, 0.16889618768328446],
        [55.735378713304854, 37.617392153477006, 0.3139425219941349],
        [55.735305929769915, 37.625065141178375, 0.3798991202346041],
        [55.7352326591313, 37.632738094752014, 0.1963542521994135],
        [55.73515889920093, 37.64041101390864, 0.20518475073313783],
        [55.73508465104372, 37.64808390219657, 0.25592023460410557],
        [55.735009915285296, 37.65575675191129, 0.2896375366568915],
        [55.73493469188522, 37.66342956750751, 0.1698017595307918],
        [55.73485897922216, 37.67110234683689, 0.26566099706744867],
        [55.7347827794848, 37.678775090670975, 0.385266862170088],
        [55.73994268710899, 37.60984222963283, 0.2538087976539589],
        [55.73987038043469, 37.61751608609462, 0.15840938416422287],
        [55.73979758603223, 37.625189907692366, 0.3086451612903226],
        [55.73972430222609, 37.63286369789862, 0.18292551319648093],
        [55.739650531258576, 37.640537451863196, 0.2761900293255132],
        [55.73957627093292, 37.648211170233, 0.3733161290322581],
        [55.739501522341506, 37.65588485374703, 0.23839530791788857],
        [55.73942628602824, 37.66355850312947, 0.2158076246334311],
        [55.73935056201643, 37.67123211627919, 0.1516809384164223],
        [55.73927434865728, 37.678905693857104, 0.3683519061583578],
        [55.739197648687465, 37.68657923665117, 0.2444715542521994],
        [55.74471875412896, 37.57926626766532, 0.18840117302052786],
        [55.744648389634186, 37.586941090849706, 0.41637067448680354],
        [55.744577535650045, 37.594615881899486, 0.48714604105571846],
        [55.744506193294995, 37.60229063874211, 0.22626627565982405],
        [55.74443436366154, 37.60996536211603, 0.30593313782991205],
        [55.74436204455265, 37.617640052669714, 0.45569970674486804],
        [55.7442892376087, 37.62531471115663, 0.20340645161290322],
        [55.74421594287896, 37.63298933266519, 1.0],
        [55.744142158688, 37.64066392066793, 0.10543108504398827],
        [55.744067887799794, 37.64833847314023, 0.1497008797653959],
        [55.743993126920934, 37.65601299069754, 0.19024046920821114],
        [55.74391787769179, 37.66368747409458, 0.2757818181818182],
        [55.74384214010788, 37.67136192403975, 0.3212480938416422],
        [55.74376591534488, 37.67903633284315, 0.5799366568914956],
        [55.743689201093964, 37.68671071238987, 0.13237771260997067],
        [55.749280362019114, 37.571710406029645, 0.16354721407624634],
        [55.74921047332905, 37.579386096395154, 0.3131683284457478],
        [55.74914009671812, 37.5870617539495, 0.1767601173020528],
        [55.74906923218227, 37.59473737940137, 0.23552375366568915],
        [55.74899787807257, 37.60241297341467, 0.3221255131964809],
        [55.74892603556024, 37.61008852829581, 0.24509560117302054],
        [55.74885370618056, 37.61776405604052, 0.13832258064516129],
        [55.748780886745784, 37.625439546024815, 0.24692551319648093],
        [55.74870757887022, 37.63311500181268, 0.4142967741935484],
        [55.74863378364606, 37.64079042414343, 0.20158592375366569],
        [55.74855949890334, 37.64846581085349, 0.2559108504398827],
        [55.7484847262826, 37.65614116269759, 0.3024516129032258],
        [55.74840946575192, 37.66381648319485, 0.24031436950146629],
        [55.748333715745055, 37.67149176457527, 0.08179706744868036],
        [55.74825747731751, 37.67916701132578, 0.9677935483870967],
        [55.7481807515893, 37.68684222137653, 0.3783741935483871],
        [55.75391042331636, 37.55647628442669, 0.7676903225806452],
        [55.7538415002487, 37.564152874407945, 0.31935718475073316],
        [55.75377208861398, 37.57182943294513, 0.267533137829912],
        [55.75370218898198, 37.57950595795046, 0.14418299120234604],
        [55.75363180134886, 37.587182450132644, 0.24789677419354839],
        [55.75356092403983, 37.59485891296671, 0.28973607038123167],
        [55.75348955932167, 37.60253533878778, 0.24317184750733137],
        [55.7534177049717, 37.61021173105558, 0.44706627565982404],
        [55.75334536263048, 37.617888090523316, 0.18874369501466276],
        [55.75327253122398, 37.62556441505847, 0.23369853372434019],
        [55.75319921239284, 37.633240705415204, 0.1256492668621701],
        [55.75312540394, 37.64091696224079, 0.5397395894428153],
        [55.75305110805391, 37.64859318630521, 0.1290791788856305],
        [55.75297632258284, 37.656269373570375, 0.20347683284457477],
        [55.752901049167534, 37.6639455247914, 0.2954322580645161],
        [55.752825287803844, 37.671621640677024, 0.2095765395894428],
        [55.75274903684278, 37.67929772188813, 0.8208797653958945],
        [55.75267229850081, 37.68697376638613, 0.47214545454545453],
        [55.752595070580895, 37.69464977481544, 0.09887624633431084],
        [55.75887102495156, 37.502850806703734, 0.6047765395894428],
        [55.75880551114125, 37.51052845284881, 0.23605395894428152],
        [55.7587395097412, 37.518206070103155, 0.24367390029325514],
        [55.75867301859498, 37.52588365443466, 0.1756574780058651],
        [55.758606039343924, 37.53356120659329, 0.26928797653958947],
        [55.75853857088782, 37.54123872725949, 0.21385102639296188],
        [55.75847061486751, 37.54891621718404, 0.2538791788856305],
        [55.75840216911113, 37.556593674207534, 0.3997466275659824],
        [55.75833323580786, 37.56427109909547, 0.17552609970674488],
        [55.75826381276078, 37.57194849249877, 0.18441290322580645],
        [55.758193901062214, 37.57962585515452, 0.14729853372434018],
        [55.75812350130774, 37.58730318216358, 0.44343460410557184],
        [55.75805261346781, 37.59498047704588, 0.19698768328445748],
        [55.757981235919694, 37.60265773765418, 0.44041290322580645],
        [55.757909370852396, 37.61033496475653, 0.15058768328445749],
        [55.75783701606888, 37.61801215900139, 0.2959390029325513],
        [55.7577641732095, 37.62568932114231, 0.18922697947214076],
        [55.75769084121816, 37.633366447172214, 0.2392492668621701],
        [55.757617021735655, 37.64104353784566, 0.33043519061583576],
        [55.757542712565076, 37.64872059380948, 0.42978064516129033],
        [55.75746791644323, 37.65639761584979, 0.20867565982404693],
        [55.75739263010399, 37.66407460177043, 0.3021325513196481],
        [55.75731685518792, 37.6717515523265, 0.15675307917888562],
        [55.75724059278703, 37.67942846825837, 0.14777712609970675],
        [55.75716384070395, 37.68710535021088, 0.2353782991202346],
        [55.763492426632915, 37.48760533027079, 0.21485982404692083],
        [55.763427880119934, 37.49528387160208, 0.30745806451612906],
        [55.763362845396365, 37.50296238260073, 0.2295131964809384],
        [55.763297320789626, 37.510640866745845, 0.15738181818181818],
        [55.76323130746448, 37.51831931633825, 0.35747565982404694],
        [55.76316480648179, 37.52599773586299, 0.10616774193548387],
        [55.76309781566919, 37.53367612316144, 0.20091964809384164],
        [55.76303033666772, 37.541354478983855, 0.16897126099706744],
        [55.76296236837703, 37.5490328040105, 0.11929149560117303],
        [55.76289391243773, 37.5567110989925, 0.01581231671554252],
        [55.76282496672861, 37.56438935614673, 0.11683753665689149],
        [55.77116570860597, 37.63374386801214, 0.04199882697947214],
        [55.776838284467, 37.503297297508624, 0.0627049853372434],
        [55.77677272756107, 37.510978284842786, 0.07856422287390029],
        [55.77573070399259, 37.62618927652915, 0.12730557184750732],
        [55.77550909816226, 37.64923057686907, 0.06816656891495601],
        [55.77528309219536, 37.67227155882731, 0.0966991202346041],
        [55.781198466342595, 37.518772615550056, 0.1186299120234604],
        [55.781131920515655, 37.52645437705956, 0.1604692082111437],
        [55.78086084823402, 37.557181111521416, 0.03962932551319648],
        [55.78065241000923, 37.58022582296823, 0.11803401759530792],
        [55.78043956885759, 37.60327023817131, 0.4584398826979472],
        [55.78014893641228, 37.633995657158444, 0.0480375366568915],
        [55.779850478387196, 37.664720520123026, 0.07463695014662756],
        [55.77977464166158, 37.672401651384845, 0.136483284457478],
        [55.779698316547616, 37.68008274325365, 0.07588035190615836],
        [55.785951580391234, 37.488155312444505, 0.11135718475073314],
        [55.78500267320341, 37.59571056696728, 0.0821208211143695],
        [55.784492269012034, 37.64948577621047, 0.17214310850439882],
        [55.78441739849473, 37.657167813282726, 0.11763049853372434],
    ]

    markers = [
        {"lat": 55.7522, "lng": 37.6156, "popup": "Test"},
    ]
    return {
        "heat_layer": heat_layer,
        "markers": markers,
    }
