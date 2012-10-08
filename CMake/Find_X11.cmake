#####################################################################
# Copyright © 2011-2012,
# Marwan Abdellah: <abdellah.marwan@gmail.com>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#####################################################################

IF(APPLE)
    # Extra X11 libraries
    SET(EXTRA_X11_LIBS "")

    # GL
    LIST(APPEND EXTRA_X11_LIBS GL)

    # X11
    LIST(APPEND EXTRA_X11_LIBS X11)

    # GLU
    LIST(APPEND EXTRA_X11_LIBS GLU)

    # glut
    LIST(APPEND EXTRA_X11_LIBS glut)

    # Linking against them
    LINK_LIBRARIES(${EXTRA_LIBS})
ENDIF(APPLE)
