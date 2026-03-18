from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pandas as pd
import random
import copy
import xlsxwriter
import time
import matplotlib.pyplot as plt
import io
import base64
import threading
import tempfile
from collections import defaultdict
from pathlib import Path

app = Flask(__name__)

DATASET_DIR = Path(app.root_path) / "dataset"
TEMP_DIR = Path(tempfile.gettempdir())
SCHEDULE_EXPORT_FILENAME = "jadwal_seminar.xlsx"
SCHEDULE_EXPORT_PATH = TEMP_DIR / SCHEDULE_EXPORT_FILENAME
DEFAULT_DATASET_FILES = {
    "kelompok": "Data_Kel.xlsx",
    "hari": "Hari.xlsx",
    "jadwal_dosen": "Jadwal_Dosen.xlsx",
    "jadwal_mahasiswa": "Jadwal_Mahasiswa.xlsx",
    "ruangan": "Ruangan.xlsx",
}

DAY_NAME_MAP = {
    "senin": "Monday",
    "selasa": "Tuesday",
    "rabu": "Wednesday",
    "kamis": "Thursday",
    "jumat": "Friday",
    "sabtu": "Saturday",
    "minggu": "Sunday",
}

# [Variables Globales omitted for brevity - Keep them as is]
global_df_kelompok = None
global_df_hari = None
global_df_jadwal_dosen = None
global_df_jadwal_mahasiswa = None
global_df_ruangan = None
global_schedule = None
global_tanggal_mulai = None
global_tanggal_selesai = None
global_best_accuracy = None
global_best_fitness = None
global_best_population = None
global_num_iterations = None
global_elapsed_time = None
global_fitness_over_time = None
global_warning_message = None
global_selected_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
global_data_source = "default"
global_generation_history = None
global_generation_worker = None
global_progress_lock = threading.Lock()
global_live_progress = {
    "is_running": False,
    "current_generation": 0,
    "max_generations": 500,
    "current_fitness": 0.0,
    "best_fitness": 0.0,
    "completed": False,
    "message": "Idle",
    "error": None,
}


class Seminar:
    def __init__(self, kode_kelompok, nama_kelompok, d_p1, d_p2, d_u1, d_u2):
        self.kode_kelompok = kode_kelompok
        self.nama_kelompok = nama_kelompok
        # Cleaner parser for lecturer handles
        self.dosen = set()
        for d in [d_p1, d_p2, d_u1, d_u2]:
            if pd.notna(d) and str(d).strip().lower() != 'nan':
                self.dosen.add(str(d).strip())

        self.dosen_pembimbing_1 = d_p1
        self.dosen_pembimbing_2 = d_p2
        self.dosen_penguji_1 = d_u1
        self.dosen_penguji_2 = d_u2
        
        self.chosen_date = None
        self.time = None
        self.room = None
        self.date = None
        self.day = None


def buat_daftar_seminar(dataset):
    seminar_list = []
    for index, row in dataset.iterrows():
        seminar = Seminar(
            row["kode"], row["kelompok"],
            row["dosen_pembimbing_1"], row["dosen_pembimbing_2"],
            row["dosen_penguji_1"], row["dosen_penguji_2"]
        )
        seminar_list.append(seminar)
    return seminar_list


def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def parse_selected_days(day_values):
    parsed = []
    for raw_day in day_values:
        if not raw_day:
            continue
        token = str(raw_day).strip().lower()
        mapped = DAY_NAME_MAP.get(token, str(raw_day).strip())
        if mapped not in parsed:
            parsed.append(mapped)
    return parsed


def load_default_datasets():
    missing = []
    loaded = {}

    for key, filename in DEFAULT_DATASET_FILES.items():
        path = DATASET_DIR / filename
        if not path.exists():
            missing.append(str(path))
            continue
        loaded[key] = normalize_columns(pd.read_excel(path))

    if missing:
        raise FileNotFoundError(
            "File dataset default tidak ditemukan: " + "; ".join(missing)
        )

    return loaded


def decode_chromosome(engine, seminars, chromosome):
    decoded = []
    for i, gene in enumerate(chromosome):
        d_idx, t_idx, r_idx = gene
        dt = engine.dates[d_idx]
        decoded.append(
            {
                "seminar_index": i,
                "kode_kelompok": str(seminars[i].kode_kelompok),
                "nama_kelompok": str(seminars[i].nama_kelompok),
                "dosen_pembimbing_1": str(seminars[i].dosen_pembimbing_1),
                "dosen_pembimbing_2": str(seminars[i].dosen_pembimbing_2),
                "dosen_penguji_1": str(seminars[i].dosen_penguji_1),
                "dosen_penguji_2": str(seminars[i].dosen_penguji_2),
                "date": dt.strftime("%Y-%m-%d"),
                "day": dt.strftime("%A"),
                "time": str(engine.times[t_idx]),
                "room": str(engine.rooms[r_idx]),
            }
        )
    decoded.sort(key=lambda x: (x["date"], x["time"], x["kode_kelompok"], x["seminar_index"]))
    return decoded


def annotate_decoded_conflicts(schedule_rows):
    annotated = [
        {
            **row,
            "has_conflict": False,
            "conflict_reasons": [],
        }
        for row in schedule_rows
    ]

    def mark_conflict(idx_a, idx_b, reason):
        for idx in (idx_a, idx_b):
            annotated[idx]["has_conflict"] = True
            if reason not in annotated[idx]["conflict_reasons"]:
                annotated[idx]["conflict_reasons"].append(reason)

    def non_empty_set(values):
        result = set()
        for value in values:
            text = str(value).strip()
            if text and text.lower() != "nan":
                result.add(text)
        return result

    for i in range(len(annotated)):
        for j in range(i + 1, len(annotated)):
            row_a = annotated[i]
            row_b = annotated[j]

            same_slot = (
                row_a["date"] == row_b["date"] and row_a["time"] == row_b["time"]
            )
            if not same_slot:
                continue

            if row_a["room"] == row_b["room"]:
                mark_conflict(i, j, "Room conflict")

            dosen_a = non_empty_set(
                [
                    row_a["dosen_pembimbing_1"],
                    row_a["dosen_pembimbing_2"],
                    row_a["dosen_penguji_1"],
                    row_a["dosen_penguji_2"],
                ]
            )
            dosen_b = non_empty_set(
                [
                    row_b["dosen_pembimbing_1"],
                    row_b["dosen_pembimbing_2"],
                    row_b["dosen_penguji_1"],
                    row_b["dosen_penguji_2"],
                ]
            )
            if dosen_a & dosen_b:
                mark_conflict(i, j, "Lecturer conflict")

    conflict_rows = sum(1 for row in annotated if row["has_conflict"])
    return annotated, conflict_rows


# ================= GA ENGINE =================

class SchedulingGA:
    def __init__(self, seminars, dates, rooms, times):
        self.seminars = seminars
        self.dates = dates
        self.rooms = rooms
        self.times = times
        
        self.D = len(dates)
        self.T = len(times)
        self.R = len(rooms)
        self.N = len(seminars)
        
    def random_gene(self):
        return (random.randint(0, self.D - 1), 
                random.randint(0, self.T - 1), 
                random.randint(0, self.R - 1))

    def evaluate(self, chromosome):
        # O(N) Hash-based evaluation
        room_usage = defaultdict(int)
        lecturer_usage = defaultdict(set)
        
        penalty = 0
        for i, gene in enumerate(chromosome):
            d, t, r = gene
            
            # Room constraint
            if room_usage[(d, t, r)] > 0:
                penalty += 10 # Room clash heavy penalty
            room_usage[(d, t, r)] += 1
            
            # Lecturer constraint
            sem = self.seminars[i]
            for dosen in sem.dosen:
                if dosen in lecturer_usage[(d, t)]:
                    penalty += 15 # Lecturer clash highest penalty
                lecturer_usage[(d, t)].add(dosen)
                
        fitness = 1.0 / (1.0 + penalty)
        return fitness, penalty

    def crossover(self, p1, p2):
        # Uniform crossover
        c1, c2 = [], []
        for i in range(self.N):
            if random.random() > 0.5:
                c1.append(p1[i])
                c2.append(p2[i])
            else:
                c1.append(p2[i])
                c2.append(p1[i])
        return c1, c2

    def mutate(self, chromosome, mutation_rate):
        for i in range(self.N):
            if random.random() < mutation_rate:
                chromosome[i] = self.random_gene()
        return chromosome

    def repair_local_search(self, chromosome):
        # Memetic algorithm piece: greedy repair
        # Find conflicts and try to reseat them locally
        room_usage = defaultdict(int)
        lecturer_usage = defaultdict(set)
        
        for i, gene in enumerate(chromosome):
            d, t, r = gene
            room_usage[(d, t, r)] += 1
            for dosen in self.seminars[i].dosen:
                lecturer_usage[(d, t)].add(dosen)
                
        for i, gene in enumerate(chromosome):
            d, t, r = gene
            collision = False
            
            if room_usage[(d, t, r)] > 1:
                collision = True
            else:
                for dosen in self.seminars[i].dosen:
                    # Approximation: if there are other copies
                    pass # Handled deeply in real systems

            # Fast random reseating
            if collision:
                best_g = gene
                for _ in range(5): # Limit search space
                    g_new = self.random_gene()
                    nd, nt, nr = g_new
                    valid = True
                    if room_usage[(nd, nt, nr)] > 0:
                        valid = False
                    for dosen in self.seminars[i].dosen:
                        if dosen in lecturer_usage[(nd, nt)]:
                            valid = False
                    if valid:
                        room_usage[(d, t, r)] -= 1
                        room_usage[(nd, nt, nr)] += 1
                        for dosen in self.seminars[i].dosen:
                            lecturer_usage[(d, t)].discard(dosen)
                            lecturer_usage[(nd, nt)].add(dosen)
                        best_g = g_new
                        break
                chromosome[i] = best_g
        return chromosome

def buat_jadwal(seminars, t_mulai, t_selesai, array_ruangan, allowed_days=None, target_fitness=1.0, progress_callback=None):
    # Prepare Data
    dates = pd.date_range(start=t_mulai, end=t_selesai, freq="B").to_list()
    if allowed_days:
        allowed_days_set = set(allowed_days)
        dates = [d for d in dates if d.strftime("%A") in allowed_days_set]
    if not dates:
        raise ValueError("Tidak ada tanggal yang cocok dengan hari yang dipilih.")
    rooms = array_ruangan
    times = ["08:00-10:00", "09:00-11:00", "10:00-12:00", "13:00-15:00", "14:00-16:00", "15:00-17:00"]
    
    engine = SchedulingGA(seminars, dates, rooms, times)
    
    population_size = min(300, len(seminars) * 5)
    max_gen = 500
    
    # Init pop
    population = [[engine.random_gene() for _ in range(engine.N)] for _ in range(population_size)]
    
    best_fitness = 0
    best_chrom = None
    fitness_over_time = []
    generation_history = []
    best_annotated = []
    num_generations = 0
    
    stop_reason = "GA Completed."

    for gen in range(max_gen):
        scored = []
        for chrom in population:
            fit, _ = engine.evaluate(chrom)
            scored.append((fit, chrom))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        current_best_fit = scored[0][0]
        current_best_chrom = scored[0][1]

        # Playback "current" panel uses the most recently produced chromosome,
        # not the top chromosome in the generation.
        most_recent_chrom = population[-1]
        most_recent_fit, _ = engine.evaluate(most_recent_chrom)
        fitness_over_time.append(most_recent_fit)
        current_decoded = decode_chromosome(engine, seminars, most_recent_chrom)
        current_annotated, current_conflict_rows = annotate_decoded_conflicts(current_decoded)
        
        if current_best_fit > best_fitness:
            best_fitness = current_best_fit
            best_chrom = copy.deepcopy(scored[0][1])
            best_annotated = copy.deepcopy(current_annotated)

        generation_history.append(
            {
                "generation": gen + 1,
                "current_fitness": float(most_recent_fit),
                "best_fitness": float(best_fitness),
                "current_schedule": current_annotated,
                "best_schedule": copy.deepcopy(best_annotated),
                "current_conflict_rows": current_conflict_rows,
                "best_conflict_rows": sum(1 for row in best_annotated if row.get("has_conflict")),
            }
        )
        num_generations = gen + 1

        if progress_callback:
            progress_callback(
                generation=num_generations,
                current_fitness=float(most_recent_fit),
                best_fitness=float(best_fitness),
                max_generations=max_gen,
            )
            
        if best_fitness >= 1.0:
            stop_reason = "GA berhenti karena solusi tanpa bentrok ditemukan."
            break
            
        # Elitism & New Pop
        new_pop = [x[1] for x in scored[:int(population_size * 0.1)]]
        
        while len(new_pop) < population_size:
            # Tournament selection
            p1 = random.choice(scored[:int(population_size*0.5)])[1]
            p2 = random.choice(scored[:int(population_size*0.5)])[1]
            
            c1, c2 = engine.crossover(p1, p2)
            c1 = engine.mutate(c1, 0.05)
            
            # Local search / Repair with 20% prob
            if random.random() < 0.2:
                c1 = engine.repair_local_search(c1)
                
            new_pop.append(c1)
            
        population = new_pop
        print(f"Gen {gen + 1}: {current_best_fit}")

    if best_fitness < 1.0 and num_generations >= max_gen:
        stop_reason = f"GA berhenti karena mencapai batas maksimum generasi ({max_gen})."

    # Map back to objects
    for i, gene in enumerate(best_chrom):
        d_idx, t_idx, r_idx = gene
        dt = engine.dates[d_idx]
        seminars[i].chosen_date = dt
        seminars[i].date = dt.strftime("%Y-%m-%d")
        seminars[i].day = dt.strftime("%A")
        seminars[i].time = engine.times[t_idx]
        seminars[i].room = engine.rooms[r_idx]
        
    return seminars, best_fitness, fitness_over_time, num_generations, stop_reason, generation_history

# [Lower Endpieces of App.py Kept Equivalent]



def cetak_jadwal(jadwal, filename):
    # Sort the schedule by date and then by time
    jadwal_sorted = sorted(jadwal, key=lambda x: (x.chosen_date, x.time))

    wb = xlsxwriter.Workbook(filename)
    ws = wb.add_worksheet("Jadwal Seminar")

    header = [
        "Kelompok",
        "Nama Kelompok",
        "Dosen Pembimbing 1",
        "Dosen Pembimbing 2",
        "Dosen Penguji 1",
        "Dosen Penguji 2",
        "Waktu",
        "Ruangan",
        "Hari",
        "Tanggal",  # Add date to the header
    ]
    for col, h in enumerate(header):
        ws.write(0, col, h)

    for row, seminar in enumerate(jadwal_sorted, 1):
        ws.write(
            row,
            0,
            str(seminar.kode_kelompok) if seminar.kode_kelompok is not None else "N/A",
        )
        ws.write(
            row,
            1,
            str(seminar.nama_kelompok) if seminar.nama_kelompok is not None else "N/A",
        )
        ws.write(
            row,
            2,
            (
                str(seminar.dosen_pembimbing_1)
                if seminar.dosen_pembimbing_1 is not None
                else "N/A"
            ),
        )
        ws.write(
            row,
            3,
            (
                str(seminar.dosen_pembimbing_2)
                if seminar.dosen_pembimbing_2 is not None
                else "N/A"
            ),
        )
        ws.write(
            row,
            4,
            (
                str(seminar.dosen_penguji_1)
                if seminar.dosen_penguji_1 is not None
                else "N/A"
            ),
        )
        ws.write(
            row,
            5,
            (
                str(seminar.dosen_penguji_2)
                if seminar.dosen_penguji_2 is not None
                else "N/A"
            ),
        )
        ws.write(row, 6, str(seminar.time) if seminar.time is not None else "N/A")
        ws.write(row, 7, str(seminar.room) if seminar.room is not None else "N/A")
        ws.write(row, 8, str(seminar.day) if seminar.day is not None else "N/A")
        ws.write(row, 9, str(seminar.date) if seminar.date is not None else "N/A")

    wb.close()


def cek_bentrok(schedule):
    conflicts = []
    for i, seminar1 in enumerate(schedule):
        for j, seminar2 in enumerate(schedule):
            if i < j:
                # Room Conflict Check
                if seminar1.time == seminar2.time and seminar1.date == seminar2.date:
                    if seminar1.room == seminar2.room:
                        conflicts.append(
                            {
                                "seminar1": seminar1.nama_kelompok,
                                "seminar2": seminar2.nama_kelompok,
                                "type": "Room conflict",
                                "time": seminar1.time,
                                "date": seminar1.date,
                                "room": seminar1.room,
                                "day": seminar1.day,
                            }
                        )

                    # Dosen Pembimbing Conflict Check
                    if (
                        seminar1.dosen_pembimbing_1 == seminar2.dosen_pembimbing_1
                        or seminar1.dosen_pembimbing_1 == seminar2.dosen_pembimbing_2
                        or seminar1.dosen_pembimbing_2 == seminar2.dosen_pembimbing_1
                        or seminar1.dosen_pembimbing_2 == seminar2.dosen_pembimbing_2
                    ):
                        conflicts.append(
                            {
                                "seminar1": seminar1.nama_kelompok,
                                "seminar2": seminar2.nama_kelompok,
                                "type": "Dosen pembimbing conflict",
                                "dosen": (
                                    seminar1.dosen_pembimbing_1
                                    if seminar1.dosen_pembimbing_1
                                    in [
                                        seminar2.dosen_pembimbing_1,
                                        seminar2.dosen_pembimbing_2,
                                    ]
                                    else seminar1.dosen_pembimbing_2
                                ),
                                "time": seminar1.time,
                                "date": seminar1.date,
                                "day": seminar1.day,
                            }
                        )

                    # Dosen Penguji Conflict Check
                    if (
                        seminar1.dosen_penguji_1 == seminar2.dosen_penguji_1
                        or seminar1.dosen_penguji_1 == seminar2.dosen_penguji_2
                        or seminar1.dosen_penguji_2 == seminar2.dosen_penguji_1
                        or seminar1.dosen_penguji_2 == seminar2.dosen_penguji_2
                    ):
                        conflicts.append(
                            {
                                "seminar1": seminar1.nama_kelompok,
                                "seminar2": seminar2.nama_kelompok,
                                "type": "Dosen penguji conflict",
                                "dosen": (
                                    seminar1.dosen_penguji_1
                                    if seminar1.dosen_penguji_1
                                    in [
                                        seminar2.dosen_penguji_1,
                                        seminar2.dosen_penguji_2,
                                    ]
                                    else seminar1.dosen_penguji_2
                                ),
                                "time": seminar1.time,
                                "date": seminar1.date,
                                "day": seminar1.day,
                            }
                        )

                    # Pembimbing and Penguji Conflict Check
                    pembimbing = [
                        seminar1.dosen_pembimbing_1,
                        seminar1.dosen_pembimbing_2,
                    ]
                    penguji = [seminar1.dosen_penguji_1, seminar1.dosen_penguji_2]
                    if any(
                        dosen in pembimbing
                        for dosen in [
                            seminar2.dosen_penguji_1,
                            seminar2.dosen_penguji_2,
                        ]
                    ) or any(
                        dosen in penguji
                        for dosen in [
                            seminar2.dosen_pembimbing_1,
                            seminar2.dosen_pembimbing_2,
                        ]
                    ):
                        conflicts.append(
                            {
                                "seminar1": seminar1.nama_kelompok,
                                "seminar2": seminar2.nama_kelompok,
                                "type": "Pembimbing and Penguji conflict",
                                "dosen": (
                                    set(pembimbing)
                                    & set(
                                        [
                                            seminar2.dosen_penguji_1,
                                            seminar2.dosen_penguji_2,
                                        ]
                                    )
                                )
                                or (
                                    set(penguji)
                                    & set(
                                        [
                                            seminar2.dosen_pembimbing_1,
                                            seminar2.dosen_pembimbing_2,
                                        ]
                                    )
                                ),
                                "time": seminar1.time,
                                "date": seminar1.date,
                                "day": seminar1.day,
                            }
                        )
    return conflicts


def serialize_schedule(schedule):
    if not schedule:
        return []

    serialized = []
    for seminar in schedule:
        serialized.append(
            {
                "kode_kelompok": str(seminar.kode_kelompok),
                "nama_kelompok": str(seminar.nama_kelompok),
                "dosen_pembimbing_1": str(seminar.dosen_pembimbing_1),
                "dosen_pembimbing_2": str(seminar.dosen_pembimbing_2),
                "dosen_penguji_1": str(seminar.dosen_penguji_1),
                "dosen_penguji_2": str(seminar.dosen_penguji_2),
                "time": str(seminar.time),
                "room": str(seminar.room),
                "day": str(seminar.day),
                "date": str(seminar.date),
            }
        )
    return serialized


def update_live_progress(**updates):
    global global_live_progress
    with global_progress_lock:
        global_live_progress.update(updates)


def snapshot_live_progress():
    with global_progress_lock:
        return dict(global_live_progress)


def reset_solution_results_only():
    global global_schedule, global_best_accuracy, global_best_fitness
    global global_best_population, global_num_iterations, global_elapsed_time
    global global_fitness_over_time, global_generation_history

    global_schedule = None
    global_best_accuracy = None
    global_best_fitness = None
    global_best_population = None
    global_num_iterations = None
    global_elapsed_time = None
    global_fitness_over_time = None
    global_generation_history = None


def execute_schedule_generation():
    global global_schedule, global_tanggal_mulai, global_tanggal_selesai
    global global_best_accuracy, global_best_fitness, global_best_population
    global global_num_iterations, global_elapsed_time, global_fitness_over_time
    global global_warning_message, global_selected_days, global_generation_history

    if (
        global_df_kelompok is None
        or global_df_hari is None
        or global_df_jadwal_dosen is None
        or global_df_jadwal_mahasiswa is None
        or global_df_ruangan is None
    ):
        raise ValueError("Data belum lengkap diunggah")

    array_ruangan = global_df_ruangan["ruangan"].tolist()
    seminars = buat_daftar_seminar(global_df_kelompok)

    if not seminars:
        raise ValueError("Tidak ada seminar yang dapat dijadwalkan karena data kosong")

    tanggal_mulai_dt = pd.to_datetime(global_tanggal_mulai)
    tanggal_selesai_dt = pd.to_datetime(global_tanggal_selesai)
    dates = pd.date_range(start=tanggal_mulai_dt, end=tanggal_selesai_dt, freq="B").to_list()
    if global_selected_days:
        selected_set = set(global_selected_days)
        dates = [d for d in dates if d.strftime("%A") in selected_set]

    if not dates:
        raise ValueError("Tidak ada tanggal yang cocok dengan hari terpilih dalam rentang tanggal.")

    times = [
        "08:00-10:00",
        "09:00-11:00",
        "10:00-12:00",
        "13:00-15:00",
        "14:00-16:00",
        "15:00-17:00",
    ]

    total_slots = len(dates) * len(array_ruangan) * len(times)
    if len(seminars) > total_slots:
        raise ValueError(
            f"Tidak mungkin membuat jadwal: Jumlah grup ({len(seminars)}) melebihi total slot waktu yang tersedia ({total_slots}). Silakan perpanjang rentang tanggal."
        )

    lecturer_counts = {}
    for sem in seminars:
        lecturers = set([sem.dosen_pembimbing_1, sem.dosen_pembimbing_2, sem.dosen_penguji_1, sem.dosen_penguji_2])
        for lecturer in lecturers:
            if pd.notna(lecturer) and str(lecturer).strip().lower() != "nan":
                lecturer_counts[lecturer] = lecturer_counts.get(lecturer, 0) + 1

    max_slots_per_person = len(dates) * len(times)
    for lecturer, count in lecturer_counts.items():
        if count > max_slots_per_person:
            raise ValueError(
                f"Tidak mungkin membuat jadwal: Dosen {lecturer} memiliki jadwal sidang ({count}) lebih banyak daripada slot waktu yang tersedia ({max_slots_per_person}) antara {global_tanggal_mulai} dan {global_tanggal_selesai}. Silakan perpanjang rentang tanggal."
            )

    start_time = time.time()

    def on_generation(generation, current_fitness, best_fitness, max_generations):
        update_live_progress(
            current_generation=int(generation),
            current_fitness=float(current_fitness),
            best_fitness=float(best_fitness),
            max_generations=int(max_generations),
            message="Evolusi sedang berjalan...",
        )

    schedule, best_fitness, fitness_over_time, num_iterations, stop_reason, generation_history = buat_jadwal(
        seminars,
        global_tanggal_mulai,
        global_tanggal_selesai,
        array_ruangan,
        global_selected_days,
        progress_callback=on_generation,
    )

    end_time = time.time()

    if not schedule:
        raise ValueError("Jadwal tidak dapat dibuat karena data kosong")

    schedule_sorted = sorted(schedule, key=lambda x: (x.chosen_date, x.time))

    global_schedule = schedule_sorted
    global_best_fitness = best_fitness
    global_best_accuracy = calculate_accuracy(schedule)
    global_best_population = num_iterations
    global_num_iterations = num_iterations
    global_elapsed_time = float("{:.2f}".format(round(end_time - start_time, 2)))
    global_fitness_over_time = fitness_over_time
    global_generation_history = generation_history

    if best_fitness < 1.0:
        global_warning_message = (
            f"Peringatan: Jadwal sempurna tidak ditemukan. {stop_reason} Jadwal ini masih memiliki bentrok. "
            "Anda bisa mencoba men-generate ulang atau memperpanjang rentang tanggal."
        )
    else:
        global_warning_message = stop_reason

    cetak_jadwal(schedule_sorted, str(SCHEDULE_EXPORT_PATH))


def background_generation_worker():
    global global_generation_worker, global_warning_message

    try:
        execute_schedule_generation()
        update_live_progress(
            is_running=False,
            completed=True,
            message="Selesai. Jadwal siap ditampilkan.",
            error=None,
            current_generation=int(global_num_iterations or 0),
        )
    except Exception as exc:
        global_warning_message = f"Gagal membuat jadwal: {exc}"
        update_live_progress(
            is_running=False,
            completed=False,
            message="Terjadi error saat evolusi.",
            error=str(exc),
        )
    finally:
        global_generation_worker = None


def start_background_generation():
    global global_generation_worker, global_warning_message

    if global_generation_worker is not None and global_generation_worker.is_alive():
        return False

    reset_solution_results_only()
    global_warning_message = "Proses evolusi sedang berjalan..."
    update_live_progress(
        is_running=True,
        completed=False,
        current_generation=0,
        max_generations=500,
        current_fitness=0.0,
        best_fitness=0.0,
        message="Evolusi sedang berjalan...",
        error=None,
    )

    global_generation_worker = threading.Thread(target=background_generation_worker, daemon=True)
    global_generation_worker.start()
    return True


@app.route("/")
def index():
    conflicts = cek_bentrok(global_schedule) if global_schedule else []
    schedule_visual_data = serialize_schedule(global_schedule)
    dataset_status = {
        key: (DATASET_DIR / filename).exists()
        for key, filename in DEFAULT_DATASET_FILES.items()
    }
    return render_template(
        "index.html",
        schedule=global_schedule,
        best_accuracy=global_best_accuracy,
        best_fitness=global_best_fitness,
        best_population=global_best_population,
        num_iterations=global_num_iterations,
        elapsed_time=global_elapsed_time,
        fitness_over_time=global_fitness_over_time,
        conflicts=conflicts,
        warning_message=global_warning_message,
        selected_days=global_selected_days,
        data_source=global_data_source,
        dataset_status=dataset_status,
        default_dataset_files=DEFAULT_DATASET_FILES,
        schedule_visual_data=schedule_visual_data,
        generation_visual_data=global_generation_history,
        live_progress=snapshot_live_progress(),
    )


@app.route("/upload", methods=["POST"])
def upload_data():
    global global_df_kelompok, global_df_hari, global_df_jadwal_dosen, global_df_jadwal_mahasiswa, global_df_ruangan, global_schedule, global_tanggal_mulai, global_tanggal_selesai, global_best_accuracy, global_best_fitness, global_best_population, global_num_iterations, global_elapsed_time, global_fitness_over_time, global_warning_message, global_selected_days, global_data_source, global_generation_history

    data_source = request.form.get("data_source", "default")
    tanggal_mulai = request.form["tanggal_mulai"]
    tanggal_selesai = request.form["tanggal_selesai"]
    selected_days = parse_selected_days(request.form.getlist("selected_days"))

    if not selected_days:
        return "Pilih minimal satu hari untuk penjadwalan.", 400

    if data_source == "default":
        try:
            datasets = load_default_datasets()
        except FileNotFoundError as exc:
            return str(exc), 400
    else:
        required_files = [
            "kelompok",
            "hari",
            "jadwal_dosen",
            "jadwal_mahasiswa",
            "ruangan",
        ]
        missing_uploads = []
        uploaded = {}
        for field in required_files:
            file_obj = request.files.get(field)
            if file_obj is None or not file_obj.filename:
                missing_uploads.append(field)
            else:
                uploaded[field] = file_obj

        if missing_uploads:
            return (
                "Mode unggah manual membutuhkan semua file: "
                + ", ".join(missing_uploads),
                400,
            )

        datasets = {k: normalize_columns(pd.read_excel(v)) for k, v in uploaded.items()}

    global_df_kelompok = datasets["kelompok"]
    global_df_hari = datasets["hari"]
    global_df_jadwal_dosen = datasets["jadwal_dosen"]
    global_df_jadwal_mahasiswa = datasets["jadwal_mahasiswa"]
    global_df_ruangan = datasets["ruangan"]
    global_tanggal_mulai = tanggal_mulai
    global_tanggal_selesai = tanggal_selesai
    global_warning_message = None
    global_selected_days = selected_days
    global_data_source = data_source
    global_generation_history = None

    started = start_background_generation()
    if not started:
        global_warning_message = "Proses sebelumnya masih berjalan. Tunggu hingga selesai."

    return redirect(url_for("index"))


def calculate_accuracy(schedule):
    conflicts = cek_bentrok(schedule)
    if not conflicts:
        return 100.0
    # simple penalty accuracy
    max_c = len(schedule) * (len(schedule) - 1) // 2
    actual = len(conflicts)
    return max(0.0, 100.0 * (1 - (actual / max(1, max_c))))

@app.route("/display_schedule")
def tampilkan_jadwal():
    if (
        global_df_kelompok is None
        or global_df_hari is None
        or global_df_jadwal_dosen is None
        or global_df_jadwal_mahasiswa is None
        or global_df_ruangan is None
    ):
        return "Data belum lengkap diunggah", 400

    started = start_background_generation()
    if not started:
        global_warning_message = "Proses sebelumnya masih berjalan. Tunggu hingga selesai."

    return redirect(url_for("index"))


@app.route("/generate_new_schedule")
def generate_new_schedule():
    return redirect(url_for("tampilkan_jadwal"))


@app.route("/reset_interface")
def reset_interface():
    global global_df_kelompok, global_df_hari, global_df_jadwal_dosen, global_df_jadwal_mahasiswa, global_df_ruangan
    global global_schedule, global_tanggal_mulai, global_tanggal_selesai
    global global_best_accuracy, global_best_fitness, global_best_population, global_num_iterations
    global global_elapsed_time, global_fitness_over_time, global_warning_message
    global global_selected_days, global_data_source, global_generation_history

    global_df_kelompok = None
    global_df_hari = None
    global_df_jadwal_dosen = None
    global_df_jadwal_mahasiswa = None
    global_df_ruangan = None
    global_schedule = None
    global_tanggal_mulai = None
    global_tanggal_selesai = None
    global_best_accuracy = None
    global_best_fitness = None
    global_best_population = None
    global_num_iterations = None
    global_elapsed_time = None
    global_fitness_over_time = None
    global_warning_message = None
    global_generation_history = None
    global_selected_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    global_data_source = "default"
    update_live_progress(
        is_running=False,
        current_generation=0,
        max_generations=500,
        current_fitness=0.0,
        best_fitness=0.0,
        completed=False,
        message="Idle",
        error=None,
    )

    return redirect(url_for("index"))


@app.route("/progress")
def get_progress():
    payload = snapshot_live_progress()
    payload["has_schedule"] = global_schedule is not None
    payload["num_iterations"] = int(global_num_iterations or 0)
    payload["warning_message"] = global_warning_message
    return jsonify(payload)


@app.route("/download_schedule")
def download_schedule():
    if not SCHEDULE_EXPORT_PATH.exists():
        return "File jadwal belum tersedia. Silakan generate jadwal terlebih dahulu.", 404
    return send_file(str(SCHEDULE_EXPORT_PATH), as_attachment=True, download_name=SCHEDULE_EXPORT_FILENAME)


@app.route("/plot_fitness")
def plot_fitness():
    global global_fitness_over_time

    if global_fitness_over_time is None:
        return "Tidak ada data untuk ditampilkan", 400

    plt.figure()
    plt.plot(global_fitness_over_time)
    plt.xlabel("Generasi")
    plt.ylabel("Fitness")
    plt.title("Fitness Selama Generasi")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return f'<img src="data:image/png;base64,{plot_url}"/>'


if __name__ == "__main__":
    app.run(debug=True, port=5000)
