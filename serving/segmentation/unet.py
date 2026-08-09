"""Schritt 3: U-Net-Architektur für binäre Lungensegmentierung (von Grund auf).

Das U-Net hat die Form eines "U": ein Encoder, der die Auflösung stufenweise
halbiert (lernt das WAS, verliert das WO), und ein spiegelbildlicher Decoder,
der sie wieder hochzieht. Die entscheidende Zutat sind die SKIP CONNECTIONS:
Jede Decoder-Stufe bekommt die gleich große Feature-Karte des Encoders direkt
angehängt und damit die feinen Kanten zurück, die das Downsampling zerstört hat.

Eingabe:  [B, 1, H, W]  (Graustufen-Röntgenbild in [0,1])
Ausgabe:  [B, 1, H, W]  ROHE LOGITS (kein Sigmoid - das macht die Loss-Funktion)

Selbsttest:  python -m segmentation.unet
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Der Grundbaustein: zweimal (Conv 3x3 -> BatchNorm -> ReLU).

    padding=1 hält die Größe konstant (bei 3x3), sodass Skip-Concats später
    ohne Zuschneiden passen. BatchNorm stabilisiert das Training von Grund auf.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Klassisches U-Net mit 4 Encoder-/Decoder-Stufen.

    base_ch steuert die Breite (Kanäle der ersten Stufe). base_ch=32 ist ein
    guter CPU-Kompromiss (~7-8 Mio. Parameter). Kanäle: 32-64-128-256, Bottleneck 512.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        c5 = base_ch * 16                       # Bottleneck

        self.pool = nn.MaxPool2d(2)             # halbiert die Auflösung

        # --- Encoder (absteigend) ---
        self.enc1 = DoubleConv(in_ch, c1)       # 256 -> Features bei 256
        self.enc2 = DoubleConv(c1, c2)          # bei 128
        self.enc3 = DoubleConv(c2, c3)          # bei 64
        self.enc4 = DoubleConv(c3, c4)          # bei 32

        # --- Bottleneck (tiefster Punkt, bei 16) ---
        self.bottleneck = DoubleConv(c4, c5)

        # --- Decoder (aufsteigend). up* verdoppelt die Auflösung (lernbar),
        #     dec* verarbeitet die Verkettung [hochskaliert | Skip]. ---
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c5, c4)          # c4 (up) + c4 (skip) = c5 Eingang
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        # --- Kopf: 1x1-Faltung auf 1 Kanal = rohe Logits pro Pixel ---
        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder: Features merken (s1..s4), das sind die Skip-Verbindungen
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        # Decoder: hochskalieren, mit passender Encoder-Karte verketten, falten
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.head(d1)                    # [B,1,H,W] Logits


if __name__ == "__main__":
    model = UNet(base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net Parameter: {n_params:,}")

    # Dummy-Vorwärtslauf: Ausgabe muss dieselbe H/W wie die Eingabe haben
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Eingabe:  {tuple(x.shape)}")
    print(f"Ausgabe:  {tuple(y.shape)}  (erwartet [2, 1, 256, 256] - gleiche H/W)")
    print(f"Ausgabe-Wertebereich: min={y.min():.2f} max={y.max():.2f}  "
          f"(rohe Logits, noch KEIN [0,1] - Sigmoid kommt in der Loss)")
