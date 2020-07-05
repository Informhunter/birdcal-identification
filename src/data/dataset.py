import os
import pandas as pd
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


class BirdTrainDataset(Dataset):
    def __init__(self, meta_path, audio_dir, target_sampling_rate):
        self.meta_path = meta_path
        self.meta_df = pd.read_csv(meta_path)
        self.audio_dir = audio_dir
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, i):
        filename = self.meta_df['filename'].values[i]
        ebird_code = self.meta_df['ebird_code'].values[i]
        filepath = os.path.join(self.audio_dir, ebird_code, filename)

        waveform, old_sampling_rate = toa.load(filepath)
        resample_transform = toa.transforms.Resample(old_sampling_rate, self.target_sampling_rate)
        waveform = resample_transform(waveform)

        channels = waveform.size(0)

        primary_label = self.meta_df['primary_label'].values[i]
        secondary_labels = literal_eval(self.meta_df['secondary_labels'].values[i])
        ebird_code = self.meta_df['ebird_code'].values[i]
        duration = self.meta_df['duration'].values[i]

        label_indices = []
        label_indices.append(EBIRD_CODE_TO_INDEX[ebird_code])
        label_indices = t.LongTensor(label_indices)

        ebird_encoded = t.zeros((len(INDEX_TO_EBIRD_CODE),))
        ebird_encoded.scatter_(0, label_indices, 1)

        return {
            'waveform': waveform,
            'old_sampling_rate': old_sampling_rate,
            'sampling_rate': self.target_sampling_rate,
            'ebird_code': ebird_code,
            'ebird_encoded': ebird_encoded,
            'primary_label': primary_label,
            'secondary_labels': secondary_labels,
            'channels': channels,
            'duration': duration,
            'filepath': filepath,
        }


class BirdTestDataset(Dataset):
    def __init__(self, meta_path, audio_dir, target_sampling_rate):
        self.meta_path = meta_path
        self.meta_df = pd.read_csv(meta_path)
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
        resample_transform = toa.transforms.Resample(old_sampling_rate, self.target_sampling_rate)
        waveform = resample_transform(waveform)
        channels = waveform.size(0)

        return {
            'waveform': waveform,
            'old_sampling_rate': old_sampling_rate,
            'sampling_rate': self.target_sampling_rate,
            'site': site,
            'audio_id': audio_id,
            'row_ids': row_ids,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'durations': durations,
            'filepath': filepath,
            'channels': channels,
        }
