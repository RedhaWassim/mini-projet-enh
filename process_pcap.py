"""
Process raw pcapng files from raw_data/ into CSV format matching the reference
dataset in data/excel_labeling_complet.csv, then merge everything into an
enriched dataset.

Usage:
    python process_pcap.py
"""

import os
import struct
import socket
import pandas as pd
from pcapng import FileScanner
from pcapng.blocks import EnhancedPacket


# ═══════════════════════════════════════════════════════════════════════
# Label Mapping: filename patterns -> canonical labels
# ═══════════════════════════════════════════════════════════════════════

LABEL_MAP = [
    # Order matters: more specific patterns first
    ("copy_54ndc47",        "Copy 54ndc47 (SMB)"),
    ("copy admin shares",   "Copy 54ndc47 (SMB)"),
    ("copy_admin_shares",   "Copy 54ndc47 (SMB)"),
    ("smb_pcapng",          "Copy 54ndc47 (SMB)"),
    ("smb)",                "Copy 54ndc47 (SMB)"),

    ("creation_services",   "Service Creation"),
    ("service creation",    "Service Creation"),
    ("service_creation",    "Service Creation"),

    ("discover_local_hosts","Discover local hosts"),
    ("discover local hosts","Discover local hosts"),

    ("view_remote",         "View remote shares"),
    ("view remote",         "View remote shares"),

    ("local_fqdn",          "BENIGNE"),
    ("local fqdn",          "BENIGNE"),
    ("host discovery",      "BENIGNE"),
]

VALID_LABELS = {
    "Copy 54ndc47 (SMB)",
    "BENIGNE",
    "Service Creation",
    "Discover local hosts",
    "View remote shares",
}


def filename_to_label(filename: str) -> str:
    """Map a pcap filename to a canonical label."""
    name_lower = filename.lower()
    for pattern, label in LABEL_MAP:
        if pattern in name_lower:
            return label
    raise ValueError(f"Cannot map filename to label: {filename}")


# ═══════════════════════════════════════════════════════════════════════
# Packet Parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_packet(raw_data: bytes, ts: float) -> dict:
    """
    Parse an Ethernet frame into the reference CSV schema:
    timestamp, src_ip, dst_ip, protocol, src_port, dst_port, length
    """
    result = {
        "timestamp": ts,
        "src_ip": "N/A",
        "dst_ip": "N/A",
        "protocol": None,
        "src_port": 0,
        "dst_port": 0,
        "length": len(raw_data),
    }

    if len(raw_data) < 14:
        return result

    # Ethernet header: 6 dst + 6 src + 2 ethertype
    eth_type = struct.unpack("!H", raw_data[12:14])[0]
    ip_offset = 14

    # Handle 802.1Q VLAN tag
    if eth_type == 0x8100:
        if len(raw_data) < 18:
            return result
        eth_type = struct.unpack("!H", raw_data[16:18])[0]
        ip_offset = 18

    # Only parse IPv4 (0x0800)
    if eth_type != 0x0800:
        return result

    if len(raw_data) < ip_offset + 20:
        return result

    # IPv4 header
    ip_header = raw_data[ip_offset:]
    ihl = (ip_header[0] & 0x0F) * 4
    protocol = ip_header[9]
    src_ip = socket.inet_ntoa(ip_header[12:16])
    dst_ip = socket.inet_ntoa(ip_header[16:20])

    result["src_ip"] = src_ip
    result["dst_ip"] = dst_ip
    result["protocol"] = protocol

    # TCP (6) or UDP (17) ports
    transport_offset = ip_offset + ihl
    if protocol in (6, 17) and len(raw_data) >= transport_offset + 4:
        result["src_port"] = struct.unpack(
            "!H", raw_data[transport_offset : transport_offset + 2]
        )[0]
        result["dst_port"] = struct.unpack(
            "!H", raw_data[transport_offset + 2 : transport_offset + 4]
        )[0]

    return result


def parse_pcapng(filepath: str) -> list[dict]:
    """Read a pcapng file and return list of packet dicts."""
    packets = []
    with open(filepath, "rb") as f:
        scanner = FileScanner(f)
        for block in scanner:
            if isinstance(block, EnhancedPacket):
                pkt = parse_packet(block.packet_data, block.timestamp)
                packets.append(pkt)
    return packets


# ═══════════════════════════════════════════════════════════════════════
# Main Processing
# ═══════════════════════════════════════════════════════════════════════

def process_all_pcaps(raw_data_dir: str) -> pd.DataFrame:
    """
    Walk through all folders in raw_data/, parse each pcapng, assign labels.
    Returns a DataFrame matching the reference schema.
    """
    all_rows = []
    folder_names = sorted(
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    )

    for folder in folder_names:
        folder_path = os.path.join(raw_data_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith((".pcapng", ".pcap"))]

        print(f"\n{'─'*60}")
        print(f"  Folder: {folder}/ ({len(files)} files)")
        print(f"{'─'*60}")

        for fname in sorted(files):
            filepath = os.path.join(folder_path, fname)

            # Map filename to label
            try:
                label = filename_to_label(fname)
            except ValueError as e:
                print(f"  [SKIP] {e}")
                continue

            print(f"  {fname}")
            print(f"    -> Label: {label}")

            # Parse packets
            try:
                packets = parse_pcapng(filepath)
            except Exception as e:
                print(f"    -> ERROR parsing: {e}")
                continue

            # Add label to each packet
            for pkt in packets:
                pkt["label"] = label

            print(f"    -> {len(packets)} packets")
            all_rows.extend(packets)

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    # Reorder columns to match reference
    df = df[["timestamp", "label", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "length"]]
    return df


def merge_with_reference(new_df: pd.DataFrame, ref_path: str, output_path: str):
    """
    Merge newly processed data with the reference CSV.
    Add packet_id, sort by timestamp, save.
    """
    print(f"\n{'='*60}")
    print(f"  MERGING DATA")
    print(f"{'='*60}")

    # Load reference
    ref_df = pd.read_csv(ref_path)
    print(f"  Reference:  {len(ref_df):>8,} packets")
    print(f"  New (pcap): {len(new_df):>8,} packets")

    # Drop packet_id from reference (will be re-assigned)
    if "packet_id" in ref_df.columns:
        ref_df = ref_df.drop(columns=["packet_id"])

    # Ensure matching dtypes
    for col in ["src_port", "dst_port", "length"]:
        new_df[col] = new_df[col].astype(int)
    new_df["protocol"] = pd.to_numeric(new_df["protocol"], errors="coerce")

    # Concatenate
    merged = pd.concat([ref_df, new_df], ignore_index=True)

    # Sort by timestamp
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Assign new packet_id
    merged.insert(0, "packet_id", range(len(merged)))

    # Validate labels
    unexpected = set(merged["label"].unique()) - VALID_LABELS
    if unexpected:
        print(f"  [WARNING] Unexpected labels found: {unexpected}")

    print(f"  Merged:     {len(merged):>8,} packets")
    print(f"\n  Label distribution:")
    for label, count in merged["label"].value_counts().items():
        pct = count / len(merged) * 100
        print(f"    {label:30s} {count:>7,}  ({pct:5.1f}%)")

    # Save
    merged.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

    return merged


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "raw_data")
    ref_path = os.path.join(base_dir, "data", "excel_labeling_complet.csv")
    output_path = os.path.join(base_dir, "data", "enriched_dataset.csv")

    # Also save the pcap-only data separately
    pcap_only_path = os.path.join(base_dir, "data", "pcap_processed.csv")

    print("=" * 60)
    print("  PCAP PROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Process all pcap files
    new_df = process_all_pcaps(raw_data_dir)

    # Save pcap-only data
    pcap_df = new_df.copy()
    pcap_df.insert(0, "packet_id", range(len(pcap_df)))
    pcap_df.to_csv(pcap_only_path, index=False)
    print(f"\n  Pcap-only data saved to: {pcap_only_path}")
    print(f"  Total new packets: {len(new_df):,}")
    print(f"\n  New data label distribution:")
    for label, count in new_df["label"].value_counts().items():
        print(f"    {label:30s} {count:>7,}")

    # Step 2: Merge with reference
    merged = merge_with_reference(new_df, ref_path, output_path)

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
