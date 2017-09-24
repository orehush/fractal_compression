# -*- coding: utf-8 -*-

import gi
gi.require_version("Gtk", "3.0")

from gi.repository import Gtk

import os
import sys
import json

from compression.algorithm import FractalCompressor


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, *relative_path)

    return os.path.join(os.path.abspath("."), *relative_path)


class Dialog(Gtk.FileChooserDialog):

    def __init__(self, title, dialog_type, buttons, file_ext=None,
                 allow_all_files=False, *args, **kwargs):

        super(Dialog, self).__init__(title, None, dialog_type, buttons,
                                     *args, **kwargs)

        if allow_all_files:
            file_filter = Gtk.FileFilter()
            file_filter.set_name("All files")
            file_filter.add_pattern("*")
            self.add_filter(file_filter)

        if file_ext:
            file_filter = Gtk.FileFilter()
            file_filter.set_name(file_ext)
            file_filter.add_pattern(file_ext)
            self.add_filter(file_filter)

        self.set_default_response(Gtk.ResponseType.OK)


class OpenDialog(Dialog):
    def __init__(self, file_ext=None, allow_all_files=False, *args, **kwargs):
        super(OpenDialog, self).__init__(
            "Open", Gtk.FileChooserAction.OPEN, (
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN, Gtk.ResponseType.OK
            ), file_ext, allow_all_files, *args, **kwargs
        )


class SaveDialog(Dialog):
    def __init__(self, title=None, file_ext=None, allow_all_files=False,
                 *args, **kwargs):
        super(SaveDialog, self).__init__(
            title or "Save", Gtk.FileChooserAction.SAVE, (
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_SAVE, Gtk.ResponseType.OK
            ), file_ext, allow_all_files, *args, **kwargs
        )


class MainWindow(object):

    def __init__(self):
        self.wTree = Gtk.Builder()
        self.wTree.add_from_file(resource_path(['templates', "main.glade"]))
        self.wTree.connect_signals(self)

        self.window = self.wTree.get_object('window1')
        self.window.set_title("Fractal compression - Rehush Dyplom Work")
        self.window.set_size_request(1200, 700)
        self.window.connect('destroy', Gtk.main_quit)
        self.window.show_all()

        self.menu_item_about = self.wTree.get_object('menu_item_about')
        self.menu_item_about.connect('activate', self.show_about_dialog)

        self.menu_item_quit = self.wTree.get_object('menu_item_quit')
        self.menu_item_quit.connect('activate', Gtk.main_quit)

        self.menu_item_open = self.wTree.get_object('menu_item_open')
        self.menu_item_open.connect('activate', self.open_image)

        self.menu_item_open_compressed = self.wTree.get_object('menu_item_open_compressed')
        self.menu_item_open_compressed.connect('activate', self.open_compressed_data)

        self.menu_item_save_decoded = self.wTree.get_object('menu_item_save_decoded')
        self.menu_item_save_decoded.connect('activate', self.save_decoded_image)

        self.menu_item_save_compressed = self.wTree.get_object('menu_item_save_compressed')
        self.menu_item_save_compressed.connect('activate', self.save_compressed_data)

        self.about_dialog = self.wTree.get_object('aboutdialog')
        self.about_dialog.connect('delete-event', self.hide_about_dialog)

        self.entry_range = self.wTree.get_object('entry_range')
        self.entry_domain = self.wTree.get_object('entry_domain')

        self.progress_bar = self.wTree.get_object('progressbar1')
        self.run_compression = self.wTree.get_object('button1')
        self.run_compression.connect('clicked', self.do_run_compression)
        self.next_iteration = self.wTree.get_object('button2')
        self.next_iteration.connect('clicked', self.do_next_iteration)
        self.quick_search = self.wTree.get_object('checkbutton1')
        self.iter_status = self.wTree.get_object('label5')

        self.image_input = self.wTree.get_object('image5')
        self.image_output = self.wTree.get_object('image6')

    def hide_about_dialog(self, *args):
        self.about_dialog.hide()
        return True

    def show_about_dialog(self, *args):
        self.about_dialog.show()

    def open_image(self, *args):
        open_dialog = OpenDialog('*.bmp')
        open_dialog.run()

        path_to_file = open_dialog.get_filename()
        open_dialog.destroy()
        if not path_to_file:
            return
        if path_to_file.split('.')[-1] != 'bmp':
            path_to_file += '.bmp'
        # self.compressor = Compressor(
        #     path_to_file, int(self.entry_range.get_text()),
        #     int(self.entry_domain.get_text()),
        # )
        self.image_input.set_from_file(path_to_file)

    def do_next_iteration(self, *args):
        # self.compressor.next_decode_iteration()
        # self.compressor.save_img('temp.bmp')
        self.image_output.set_from_file('temp.bmp')
        # self.iter_status.set_label('Number of iterations: %s' % self.compressor.iter_count)

    def callback(self, number_iter):
        self.progress_bar.set_fraction(
            number_iter #/
            # (self.compressor.shape[0]/float(self.entry_range.get_text())**2)
        )

    def do_run_compression(self, *args):
        if self.quick_search.get_active():
            pass
            # self.compressor.encode_with_ternary_tree(self.callback)
        else:
            pass
            # self.compressor.encode(self.callback)

    def open_compressed_data(self, *args):
        open_dialog = OpenDialog('*.json')
        open_dialog.run()

        path_to_file = open_dialog.get_filename()
        open_dialog.destroy()
        if not path_to_file:
            return
        if path_to_file.split('.')[-1] != 'json':
            path_to_file += '.json'
        # self.compressor = Compressor.create_from_compressed_file(path_to_file)

    def save_compressed_data(self, *args):
        save_dialog = SaveDialog('*.json')
        save_dialog.run()

        path_to_file = save_dialog.get_filename()
        save_dialog.destroy()

        if not path_to_file:
            return
        if path_to_file.split('.')[-1] != 'json':
            path_to_file += '.json'
        # self.compressor.save_transformation(path_to_file)

    def save_decoded_image(self, *args):
        save_dialog = SaveDialog('*.bmp')
        save_dialog.run()

        path_to_file = save_dialog.get_filename()
        save_dialog.destroy()

        if not path_to_file:
            return
        if path_to_file.split('.')[-1] != 'bmp':
            path_to_file += '.bmp'
        # self.compressor.save_img(path_to_file)


if __name__ == '__main__':
    app = MainWindow()
    Gtk.main()
