# -*- mode: python -*-

block_cipher = None


a = Analysis(['gui/main_window.py'],
             pathex=['/home/orehush/Studing/Dyplom/practice'],
             binaries=None,
             datas=[('/home/orehush/Studing/Dyplom/practice/gui/templates/*', 'templates')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='main_window',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main_window')
