'''
Created on Nov 4, 2012

@author: Erics
'''


def runGames( layout, pacman, ghosts, display, numGames, record, numTraining = 0, catchExceptions=False, timeout=30 ):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []

    for i in range( numGames ):
        beQuiet = i < numTraining
        if beQuiet:
                # Suppress output and graphics
                import textDisplay
                gameDisplay = textDisplay.NullGraphics()
                rules.quiet = True
        else:
                gameDisplay = display
                rules.quiet = False
        game = rules.newGame( layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions)
        game.run()
        if not beQuiet: games.append(game)

        if record:
            import time, cPickle
            fname = ('recorded-game-%d' % (i + 1)) +    '-'.join([str(t) for t in time.localtime()[1:6]])
            f = file(fname, 'w')
            components = {'layout': layout, 'actions': game.moveHistory}
            cPickle.dump(components, f)
            f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True)/ float(len(wins))
        print 'Average Score:', sum(scores) / float(len(scores))
        print 'Scores:             ', ', '.join([str(score) for score in scores])
        print 'Win Rate:            %d/%d (%.2f)' % (wins.count(True), len(wins), winRate)
        print 'Record:             ', ', '.join([ ['Loss', 'Win'][int(w)] for w in wins])

    return games




if __name__ == '__main__':
    sys.argv = raw_input('Enter command line arguments: ').split()
    args = readCommand( sys.argv ) # Get game components based on input
    print args
    runGames( **args )
    run()
    pass