from graphviz import Graph

e = Graph('ER', filename='er.gv', engine='neato')
e.node('h1')
e.node('h2')
