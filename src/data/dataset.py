import os
from torch.utils.data import Dataset
import torch as t
import torchaudio as toa
from ast import literal_eval


INDEX_TO_EBIRD_CODE = [
    'aldfly', 'ameavo', 'amebit', 'amecro', 'amegfi', 'amekes',
    'amepip', 'amered', 'amerob', 'amewig', 'amewoo', 'amtspa',
    'annhum', 'astfly', 'baisan', 'baleag', 'balori', 'banswa',
    'barswa', 'bawwar', 'belkin1', 'belspa2', 'bewwre', 'bkbcuc',
    'bkbmag1', 'bkbwar', 'bkcchi', 'bkchum', 'bkhgro', 'bkpwar',
    'bktspa', 'blkpho', 'blugrb1', 'blujay', 'bnhcow', 'boboli',
    'bongul', 'brdowl', 'brebla', 'brespa', 'brncre', 'brnthr',
    'brthum', 'brwhaw', 'btbwar', 'btnwar', 'btywar', 'buffle',
    'buggna', 'buhvir', 'bulori', 'bushti', 'buwtea', 'buwwar',
    'cacwre', 'calgul', 'calqua', 'camwar', 'cangoo', 'canwar',
    'canwre', 'carwre', 'casfin', 'caster1', 'casvir', 'cedwax',
    'chispa', 'chiswi', 'chswar', 'chukar', 'clanut', 'cliswa',
    'comgol', 'comgra', 'comloo', 'commer', 'comnig', 'comrav',
    'comred', 'comter', 'comyel', 'coohaw', 'coshum', 'cowscj1',
    'daejun', 'doccor', 'dowwoo', 'dusfly', 'eargre', 'easblu',
    'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eucdov',
    'eursta', 'evegro', 'fiespa', 'fiscro', 'foxspa', 'gadwal',
    'gcrfin', 'gnttow', 'gnwtea', 'gockin', 'gocspa', 'goleag',
    'grbher3', 'grcfly', 'greegr', 'greroa', 'greyel', 'grhowl',
    'grnher', 'grtgra', 'grycat', 'gryfly', 'haiwoo', 'hamfly',
    'hergul', 'herthr', 'hoomer', 'hoowar', 'horgre', 'horlar',
    'houfin', 'houspa', 'houwre', 'indbun', 'juntit1', 'killde',
    'labwoo', 'larspa', 'lazbun', 'leabit', 'leafly', 'leasan',
    'lecthr', 'lesgol', 'lesnig', 'lesyel', 'lewwoo', 'linspa',
    'lobcur', 'lobdow', 'logshr', 'lotduc', 'louwat', 'macwar',
    'magwar', 'mallar3', 'marwre', 'merlin', 'moublu', 'mouchi',
    'moudov', 'norcar', 'norfli', 'norhar2', 'normoc', 'norpar',
    'norpin', 'norsho', 'norwat', 'nrwswa', 'nutwoo', 'olsfly',
    'orcwar', 'osprey', 'ovenbi1', 'palwar', 'pasfly', 'pecsan',
    'perfal', 'phaino', 'pibgre', 'pilwoo', 'pingro', 'pinjay',
    'pinsis', 'pinwar', 'plsvir', 'prawar', 'purfin', 'pygnut',
    'rebmer', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'redhea',
    'reevir1', 'renpha', 'reshaw', 'rethaw', 'rewbla', 'ribgul',
    'rinduc', 'robgro', 'rocpig', 'rocwre', 'rthhum', 'ruckin',
    'rudduc', 'rufgro', 'rufhum', 'rusbla', 'sagspa1', 'sagthr',
    'savspa', 'saypho', 'scatan', 'scoori', 'semplo', 'semsan',
    'sheowl', 'shshaw', 'snobun', 'snogoo', 'solsan', 'sonspa',
    'sora', 'sposan', 'spotow', 'stejay', 'swahaw', 'swaspa',
    'swathr', 'treswa', 'truswa', 'tuftit', 'tunswa', 'veery',
    'vesspa', 'vigswa', 'warvir', 'wesblu', 'wesgre', 'weskin',
    'wesmea', 'wessan', 'westan', 'wewpew', 'whbnut', 'whcspa',
    'whfibi', 'whtspa', 'whtswi', 'wilfly', 'wilsni1', 'wiltur',
    'winwre3', 'wlswar', 'wooduc', 'wooscj2', 'woothr', 'y00475',
    'yebfly', 'yebsap', 'yehbla', 'yelwar', 'yerwar', 'yetvir'
]


EBIRD_CODE_TO_INDEX = {code: index for index, code in enumerate(INDEX_TO_EBIRD_CODE)}


EBIRD_CODE_TO_LABEL = {
    'aldfly': 'Empidonax alnorum_Alder Flycatcher',
    'ameavo': 'Recurvirostra americana_American Avocet',
    'amebit': 'Botaurus lentiginosus_American Bittern',
    'amecro': 'Corvus brachyrhynchos_American Crow',
    'amegfi': 'Spinus tristis_American Goldfinch',
    'amekes': 'Falco sparverius_American Kestrel',
    'amepip': 'Anthus rubescens_American Pipit',
    'amered': 'Setophaga ruticilla_American Redstart',
    'amerob': 'Turdus migratorius_American Robin',
    'amewig': 'Mareca americana_American Wigeon',
    'amewoo': 'Scolopax minor_American Woodcock',
    'amtspa': 'Spizelloides arborea_American Tree Sparrow',
    'annhum': "Calypte anna_Anna's Hummingbird",
    'astfly': 'Myiarchus cinerascens_Ash-throated Flycatcher',
    'baisan': "Calidris bairdii_Baird's Sandpiper",
    'baleag': 'Haliaeetus leucocephalus_Bald Eagle',
    'balori': 'Icterus galbula_Baltimore Oriole',
    'banswa': 'Riparia riparia_Bank Swallow',
    'barswa': 'Hirundo rustica_Barn Swallow',
    'bawwar': 'Mniotilta varia_Black-and-white Warbler',
    'belkin1': 'Megaceryle alcyon_Belted Kingfisher',
    'belspa2': "Artemisiospiza belli_Bell's Sparrow",
    'bewwre': "Thryomanes bewickii_Bewick's Wren",
    'bkbcuc': 'Coccyzus erythropthalmus_Black-billed Cuckoo',
    'bkbmag1': 'Pica hudsonia_Black-billed Magpie',
    'bkbwar': 'Setophaga fusca_Blackburnian Warbler',
    'bkcchi': 'Poecile atricapillus_Black-capped Chickadee',
    'bkchum': 'Archilochus alexandri_Black-chinned Hummingbird',
    'bkhgro': 'Pheucticus melanocephalus_Black-headed Grosbeak',
    'bkpwar': 'Setophaga striata_Blackpoll Warbler',
    'bktspa': 'Amphispiza bilineata_Black-throated Sparrow',
    'blkpho': 'Sayornis nigricans_Black Phoebe',
    'blugrb1': 'Passerina caerulea_Blue Grosbeak',
    'blujay': 'Cyanocitta cristata_Blue Jay',
    'bnhcow': 'Molothrus ater_Brown-headed Cowbird',
    'boboli': 'Dolichonyx oryzivorus_Bobolink',
    'bongul': "Chroicocephalus philadelphia_Bonaparte's Gull",
    'brdowl': 'Strix varia_Barred Owl',
    'brebla': "Euphagus cyanocephalus_Brewer's Blackbird",
    'brespa': "Spizella breweri_Brewer's Sparrow",
    'brncre': 'Certhia americana_Brown Creeper',
    'brnthr': 'Toxostoma rufum_Brown Thrasher',
    'brthum': 'Selasphorus platycercus_Broad-tailed Hummingbird',
    'brwhaw': 'Buteo platypterus_Broad-winged Hawk',
    'btbwar': 'Setophaga caerulescens_Black-throated Blue Warbler',
    'btnwar': 'Setophaga virens_Black-throated Green Warbler',
    'btywar': 'Setophaga nigrescens_Black-throated Gray Warbler',
    'buffle': 'Bucephala albeola_Bufflehead',
    'buggna': 'Polioptila caerulea_Blue-gray Gnatcatcher',
    'buhvir': 'Vireo solitarius_Blue-headed Vireo',
    'bulori': "Icterus bullockii_Bullock's Oriole",
    'bushti': 'Psaltriparus minimus_Bushtit',
    'buwtea': 'Spatula discors_Blue-winged Teal',
    'buwwar': 'Vermivora cyanoptera_Blue-winged Warbler',
    'cacwre': 'Campylorhynchus brunneicapillus_Cactus Wren',
    'calgul': 'Larus californicus_California Gull',
    'calqua': 'Callipepla californica_California Quail',
    'camwar': 'Setophaga tigrina_Cape May Warbler',
    'cangoo': 'Branta canadensis_Canada Goose',
    'canwar': 'Cardellina canadensis_Canada Warbler',
    'canwre': 'Catherpes mexicanus_Canyon Wren',
    'carwre': 'Thryothorus ludovicianus_Carolina Wren',
    'casfin': "Haemorhous cassinii_Cassin's Finch",
    'caster1': 'Hydroprogne caspia_Caspian Tern',
    'casvir': "Vireo cassinii_Cassin's Vireo",
    'cedwax': 'Bombycilla cedrorum_Cedar Waxwing',
    'chispa': 'Spizella passerina_Chipping Sparrow',
    'chiswi': 'Chaetura pelagica_Chimney Swift',
    'chswar': 'Setophaga pensylvanica_Chestnut-sided Warbler',
    'chukar': 'Alectoris chukar_Chukar',
    'clanut': "Nucifraga columbiana_Clark's Nutcracker",
    'cliswa': 'Petrochelidon pyrrhonota_Cliff Swallow',
    'comgol': 'Bucephala clangula_Common Goldeneye',
    'comgra': 'Quiscalus quiscula_Common Grackle',
    'comloo': 'Gavia immer_Common Loon',
    'commer': 'Mergus merganser_Common Merganser',
    'comnig': 'Chordeiles minor_Common Nighthawk',
    'comrav': 'Corvus corax_Common Raven',
    'comred': 'Acanthis flammea_Common Redpoll',
    'comter': 'Sterna hirundo_Common Tern',
    'comyel': 'Geothlypis trichas_Common Yellowthroat',
    'coohaw': "Accipiter cooperii_Cooper's Hawk",
    'coshum': "Calypte costae_Costa's Hummingbird",
    'cowscj1': 'Aphelocoma californica_California Scrub-Jay',
    'daejun': 'Junco hyemalis_Dark-eyed Junco',
    'doccor': 'Phalacrocorax auritus_Double-crested Cormorant',
    'dowwoo': 'Dryobates pubescens_Downy Woodpecker',
    'dusfly': 'Empidonax oberholseri_Dusky Flycatcher',
    'eargre': 'Podiceps nigricollis_Eared Grebe',
    'easblu': 'Sialia sialis_Eastern Bluebird',
    'easkin': 'Tyrannus tyrannus_Eastern Kingbird',
    'easmea': 'Sturnella magna_Eastern Meadowlark',
    'easpho': 'Sayornis phoebe_Eastern Phoebe',
    'eastow': 'Pipilo erythrophthalmus_Eastern Towhee',
    'eawpew': 'Contopus virens_Eastern Wood-Pewee',
    'eucdov': 'Streptopelia decaocto_Eurasian Collared-Dove',
    'eursta': 'Sturnus vulgaris_European Starling',
    'evegro': 'Coccothraustes vespertinus_Evening Grosbeak',
    'fiespa': 'Spizella pusilla_Field Sparrow',
    'fiscro': 'Corvus ossifragus_Fish Crow',
    'foxspa': 'Passerella iliaca_Fox Sparrow',
    'gadwal': 'Mareca strepera_Gadwall',
    'gcrfin': 'Leucosticte tephrocotis_Gray-crowned Rosy-Finch',
    'gnttow': 'Pipilo chlorurus_Green-tailed Towhee',
    'gnwtea': 'Anas crecca_Green-winged Teal',
    'gockin': 'Regulus satrapa_Golden-crowned Kinglet',
    'gocspa': 'Zonotrichia atricapilla_Golden-crowned Sparrow',
    'goleag': 'Aquila chrysaetos_Golden Eagle',
    'grbher3': 'Ardea herodias_Great Blue Heron',
    'grcfly': 'Myiarchus crinitus_Great Crested Flycatcher',
    'greegr': 'Ardea alba_Great Egret',
    'greroa': 'Geococcyx californianus_Greater Roadrunner',
    'greyel': 'Tringa melanoleuca_Greater Yellowlegs',
    'grhowl': 'Bubo virginianus_Great Horned Owl',
    'grnher': 'Butorides virescens_Green Heron',
    'grtgra': 'Quiscalus mexicanus_Great-tailed Grackle',
    'grycat': 'Dumetella carolinensis_Gray Catbird',
    'gryfly': 'Empidonax wrightii_Gray Flycatcher',
    'haiwoo': 'Dryobates villosus_Hairy Woodpecker',
    'hamfly': "Empidonax hammondii_Hammond's Flycatcher",
    'hergul': 'Larus argentatus_Herring Gull',
    'herthr': 'Catharus guttatus_Hermit Thrush',
    'hoomer': 'Lophodytes cucullatus_Hooded Merganser',
    'hoowar': 'Setophaga citrina_Hooded Warbler',
    'horgre': 'Podiceps auritus_Horned Grebe',
    'horlar': 'Eremophila alpestris_Horned Lark',
    'houfin': 'Haemorhous mexicanus_House Finch',
    'houspa': 'Passer domesticus_House Sparrow',
    'houwre': 'Troglodytes aedon_House Wren',
    'indbun': 'Passerina cyanea_Indigo Bunting',
    'juntit1': 'Baeolophus ridgwayi_Juniper Titmouse',
    'killde': 'Charadrius vociferus_Killdeer',
    'labwoo': 'Dryobates scalaris_Ladder-backed Woodpecker',
    'larspa': 'Chondestes grammacus_Lark Sparrow',
    'lazbun': 'Passerina amoena_Lazuli Bunting',
    'leabit': 'Ixobrychus exilis_Least Bittern',
    'leafly': 'Empidonax minimus_Least Flycatcher',
    'leasan': 'Calidris minutilla_Least Sandpiper',
    'lecthr': "Toxostoma lecontei_LeConte's Thrasher",
    'lesgol': 'Spinus psaltria_Lesser Goldfinch',
    'lesnig': 'Chordeiles acutipennis_Lesser Nighthawk',
    'lesyel': 'Tringa flavipes_Lesser Yellowlegs',
    'lewwoo': "Melanerpes lewis_Lewis's Woodpecker",
    'linspa': "Melospiza lincolnii_Lincoln's Sparrow",
    'lobcur': 'Numenius americanus_Long-billed Curlew',
    'lobdow': 'Limnodromus scolopaceus_Long-billed Dowitcher',
    'logshr': 'Lanius ludovicianus_Loggerhead Shrike',
    'lotduc': 'Clangula hyemalis_Long-tailed Duck',
    'louwat': 'Parkesia motacilla_Louisiana Waterthrush',
    'macwar': "Geothlypis tolmiei_MacGillivray's Warbler",
    'magwar': 'Setophaga magnolia_Magnolia Warbler',
    'mallar3': 'Anas platyrhynchos_Mallard',
    'marwre': 'Cistothorus palustris_Marsh Wren',
    'merlin': 'Falco columbarius_Merlin',
    'moublu': 'Sialia currucoides_Mountain Bluebird',
    'mouchi': 'Poecile gambeli_Mountain Chickadee',
    'moudov': 'Zenaida macroura_Mourning Dove',
    'norcar': 'Cardinalis cardinalis_Northern Cardinal',
    'norfli': 'Colaptes auratus_Northern Flicker',
    'norhar2': 'Circus hudsonius_Northern Harrier',
    'normoc': 'Mimus polyglottos_Northern Mockingbird',
    'norpar': 'Setophaga americana_Northern Parula',
    'norpin': 'Anas acuta_Northern Pintail',
    'norsho': 'Spatula clypeata_Northern Shoveler',
    'norwat': 'Parkesia noveboracensis_Northern Waterthrush',
    'nrwswa': 'Stelgidopteryx serripennis_Northern Rough-winged Swallow',
    'nutwoo': "Dryobates nuttallii_Nuttall's Woodpecker",
    'olsfly': 'Contopus cooperi_Olive-sided Flycatcher',
    'orcwar': 'Leiothlypis celata_Orange-crowned Warbler',
    'osprey': 'Pandion haliaetus_Osprey',
    'ovenbi1': 'Seiurus aurocapilla_Ovenbird',
    'palwar': 'Setophaga palmarum_Palm Warbler',
    'pasfly': 'Empidonax difficilis_Pacific-slope Flycatcher',
    'pecsan': 'Calidris melanotos_Pectoral Sandpiper',
    'perfal': 'Falco peregrinus_Peregrine Falcon',
    'phaino': 'Phainopepla nitens_Phainopepla',
    'pibgre': 'Podilymbus podiceps_Pied-billed Grebe',
    'pilwoo': 'Dryocopus pileatus_Pileated Woodpecker',
    'pingro': 'Pinicola enucleator_Pine Grosbeak',
    'pinjay': 'Gymnorhinus cyanocephalus_Pinyon Jay',
    'pinsis': 'Spinus pinus_Pine Siskin',
    'pinwar': 'Setophaga pinus_Pine Warbler',
    'plsvir': 'Vireo plumbeus_Plumbeous Vireo',
    'prawar': 'Setophaga discolor_Prairie Warbler',
    'purfin': 'Haemorhous purpureus_Purple Finch',
    'pygnut': 'Sitta pygmaea_Pygmy Nuthatch',
    'rebmer': 'Mergus serrator_Red-breasted Merganser',
    'rebnut': 'Sitta canadensis_Red-breasted Nuthatch',
    'rebsap': 'Sphyrapicus ruber_Red-breasted Sapsucker',
    'rebwoo': 'Melanerpes carolinus_Red-bellied Woodpecker',
    'redcro': 'Loxia curvirostra_Red Crossbill',
    'redhea': 'Aythya americana_Redhead',
    'reevir1': 'Vireo olivaceus_Red-eyed Vireo',
    'renpha': 'Phalaropus lobatus_Red-necked Phalarope',
    'reshaw': 'Buteo lineatus_Red-shouldered Hawk',
    'rethaw': 'Buteo jamaicensis_Red-tailed Hawk',
    'rewbla': 'Agelaius phoeniceus_Red-winged Blackbird',
    'ribgul': 'Larus delawarensis_Ring-billed Gull',
    'rinduc': 'Aythya collaris_Ring-necked Duck',
    'robgro': 'Pheucticus ludovicianus_Rose-breasted Grosbeak',
    'rocpig': 'Columba livia_Rock Pigeon',
    'rocwre': 'Salpinctes obsoletus_Rock Wren',
    'rthhum': 'Archilochus colubris_Ruby-throated Hummingbird',
    'ruckin': 'Regulus calendula_Ruby-crowned Kinglet',
    'rudduc': 'Oxyura jamaicensis_Ruddy Duck',
    'rufgro': 'Bonasa umbellus_Ruffed Grouse',
    'rufhum': 'Selasphorus rufus_Rufous Hummingbird',
    'rusbla': 'Euphagus carolinus_Rusty Blackbird',
    'sagspa1': 'Artemisiospiza nevadensis_Sagebrush Sparrow',
    'sagthr': 'Oreoscoptes montanus_Sage Thrasher',
    'savspa': 'Passerculus sandwichensis_Savannah Sparrow',
    'saypho': "Sayornis saya_Say's Phoebe",
    'scatan': 'Piranga olivacea_Scarlet Tanager',
    'scoori': "Icterus parisorum_Scott's Oriole",
    'semplo': 'Charadrius semipalmatus_Semipalmated Plover',
    'semsan': 'Calidris pusilla_Semipalmated Sandpiper',
    'sheowl': 'Asio flammeus_Short-eared Owl',
    'shshaw': 'Accipiter striatus_Sharp-shinned Hawk',
    'snobun': 'Plectrophenax nivalis_Snow Bunting',
    'snogoo': 'Anser caerulescens_Snow Goose',
    'solsan': 'Tringa solitaria_Solitary Sandpiper',
    'sonspa': 'Melospiza melodia_Song Sparrow',
    'sora': 'Porzana carolina_Sora',
    'sposan': 'Actitis macularius_Spotted Sandpiper',
    'spotow': 'Pipilo maculatus_Spotted Towhee',
    'stejay': "Cyanocitta stelleri_Steller's Jay",
    'swahaw': "Buteo swainsoni_Swainson's Hawk",
    'swaspa': 'Melospiza georgiana_Swamp Sparrow',
    'swathr': "Catharus ustulatus_Swainson's Thrush",
    'treswa': 'Tachycineta bicolor_Tree Swallow',
    'truswa': 'Cygnus buccinator_Trumpeter Swan',
    'tuftit': 'Baeolophus bicolor_Tufted Titmouse',
    'tunswa': 'Cygnus columbianus_Tundra Swan',
    'veery': 'Catharus fuscescens_Veery',
    'vesspa': 'Pooecetes gramineus_Vesper Sparrow',
    'vigswa': 'Tachycineta thalassina_Violet-green Swallow',
    'warvir': 'Vireo gilvus_Warbling Vireo',
    'wesblu': 'Sialia mexicana_Western Bluebird',
    'wesgre': 'Aechmophorus occidentalis_Western Grebe',
    'weskin': 'Tyrannus verticalis_Western Kingbird',
    'wesmea': 'Sturnella neglecta_Western Meadowlark',
    'wessan': 'Calidris mauri_Western Sandpiper',
    'westan': 'Piranga ludoviciana_Western Tanager',
    'wewpew': 'Contopus sordidulus_Western Wood-Pewee',
    'whbnut': 'Sitta carolinensis_White-breasted Nuthatch',
    'whcspa': 'Zonotrichia leucophrys_White-crowned Sparrow',
    'whfibi': 'Plegadis chihi_White-faced Ibis',
    'whtspa': 'Zonotrichia albicollis_White-throated Sparrow',
    'whtswi': 'Aeronautes saxatalis_White-throated Swift',
    'wilfly': 'Empidonax traillii_Willow Flycatcher',
    'wilsni1': "Gallinago delicata_Wilson's Snipe",
    'wiltur': 'Meleagris gallopavo_Wild Turkey',
    'winwre3': 'Troglodytes hiemalis_Winter Wren',
    'wlswar': "Cardellina pusilla_Wilson's Warbler",
    'wooduc': 'Aix sponsa_Wood Duck',
    'wooscj2': "Aphelocoma woodhouseii_Woodhouse's Scrub-Jay",
    'woothr': 'Hylocichla mustelina_Wood Thrush',
    'y00475': 'Fulica americana_American Coot',
    'yebfly': 'Empidonax flaviventris_Yellow-bellied Flycatcher',
    'yebsap': 'Sphyrapicus varius_Yellow-bellied Sapsucker',
    'yehbla': 'Xanthocephalus xanthocephalus_Yellow-headed Blackbird',
    'yelwar': 'Setophaga petechia_Yellow Warbler',
    'yerwar': 'Setophaga coronata_Yellow-rumped Warbler',
    'yetvir': 'Vireo flavifrons_Yellow-throated Vireo'
}


LABEL_TO_EBIRD_CODE = {v: k for k, v in EBIRD_CODE_TO_LABEL.items()}


class BirdMelTrainDataset(Dataset):
    '''Mel spectrogram train dataset.'''
    def __init__(self, meta_df, mels_dir, encode_secondary_labels, transform=None):
        self.meta_df = meta_df
        self.mels_dir = mels_dir
        self.encode_secondary_labels = encode_secondary_labels
        self.transform = transform

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, i):
        filename = os.path.splitext(self.meta_df['filename'].values[i])[0] + '.pt'
        primary_ebird_code = self.meta_df['ebird_code'].values[i]
        filepath = os.path.join(self.mels_dir, primary_ebird_code, filename)

        mel_spec = t.load(filepath)

        primary_label = self.meta_df['primary_label'].values[i]
        secondary_labels = literal_eval(self.meta_df['secondary_labels'].values[i])
        duration = self.meta_df['duration'].values[i]
        rating = self.meta_df['rating'].values[i]

        secondary_ebird_codes = []
        for secondary_label in secondary_labels:
            if secondary_label in LABEL_TO_EBIRD_CODE:
                secondary_ebird_code = LABEL_TO_EBIRD_CODE[secondary_label]
                secondary_ebird_codes.append(secondary_ebird_code)

        primary_label_indices = []
        primary_label_indices.append(EBIRD_CODE_TO_INDEX[primary_ebird_code])
        primary_label_indices = t.LongTensor(primary_label_indices)

        encoded_ebird_codes = t.zeros((len(INDEX_TO_EBIRD_CODE),))
        encoded_ebird_codes.scatter_(0, primary_label_indices, 1)

        if self.encode_secondary_labels:
            secondary_label_indices = []
            for secondary_ebird_code in secondary_ebird_codes:
                secondary_label_indices.append(EBIRD_CODE_TO_INDEX[secondary_ebird_code])
            secondary_label_indices = t.LongTensor(secondary_label_indices)
            encoded_ebird_codes.scatter_(0, secondary_label_indices, 1)

        sample = {
            'mel_spec': mel_spec,
            'primary_ebird_code': primary_ebird_code,
            'secondary_ebird_codes': secondary_ebird_codes,
            'encoded_ebird_codes': encoded_ebird_codes,
            'primary_label': primary_label,
            'secondary_labels': secondary_labels,
            'duration': duration,
            'filepath': filepath,
            'rating': rating,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class BirdMelTestDataset(Dataset):
    '''Test audio dataset with mel spectrogram transformation.'''
    def __init__(self, meta_df, audio_dir, target_sampling_rate=None):
        self.meta_df = meta_df
        self.audio_dir = audio_dir
        self.audio_ids = self.meta_df['audio_id'].unique()
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, i):

        audio_id = self.audio_ids[i]

        df = self.meta_df[self.meta_df['audio_id'] == audio_id]
        site = df['site'].values[0]
        row_ids = list(df['row_id'].values)

        start_seconds = []
        end_seconds = []
        durations = []
        current_seconds = 0
        for seconds in df['seconds'].values:
            start_seconds.append(current_seconds)
            end_seconds.append(seconds)
            durations.append(seconds - current_seconds)
            current_seconds = seconds

        filepath = os.path.join(self.audio_dir, f'{audio_id}.mp3')

        waveform, old_sampling_rate = toa.load(filepath)
        if self.target_sampling_rate is not None:
            resample_transform = toa.transforms.Resample(
                old_sampling_rate,
                self.target_sampling_rate
            )
            waveform = resample_transform(waveform)
            sampling_rate = self.target_sampling_rate
        else:
            sampling_rate = old_sampling_rate

        channels = waveform.size(0)

        mel_transform = toa.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=2048,
            n_mels=128,
            hop_length=512,
        )

        mel_spec = mel_transform(waveform)

        return {
            'waveform': waveform,
            'mel_spec': mel_spec,
            'old_sampling_rate': old_sampling_rate,
            'sampling_rate': sampling_rate,
            'site': site,
            'audio_id': audio_id,
            'row_ids': row_ids,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'durations': durations,
            'filepath': filepath,
            'channels': channels,
        }
