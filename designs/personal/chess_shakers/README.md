# Chess Salt & Pepper Shakers

King (salt) and Queen (pepper) chess piece shakers with a two-square chessboard stand.

**Features**
- Fully hollow figures with detailed surface relief (robe folds, crowns, jewelry)
- Eccentric-circle printed threads — no hardware, no supports
- Two-tone ready: dark square on stand prints at a different height for filament swap
- Salt holes on crown dome (8 × 1.4 mm); pepper holes on back (7 × 2.2 mm)
- Coin-slot plug seals the fill opening

**Print settings**
- Layer height: 0.15 mm
- No supports needed (designed to print upright)
- Filament change at layer corresponding to `DARK_TOP = 4.4 mm` for two-color board
- Bed: fits Bambu A1 mini (180 mm) — king 47 mm diameter, queen slightly smaller

**Dependencies**
```
pip install trimesh manifold3d numpy matplotlib pillow pyembree
```

**Usage**
```bash
python chess_shakers.py        # outputs king_salt_shaker.stl, queen_pepper_shaker.stl,
                               #         chessboard_stand.stl, shaker_fill_plug.stl
```
